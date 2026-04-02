{
  description = "Harmony — decentralized internet stack";

  inputs = {
    # Unstable channel — needed for Rust 1.85+ (edition2024 in Cargo.lock).
    # nixos-24.11 shipped Rust 1.82 which is too old for cranelift-codegen-shared.
    # Currently provides Rust 1.94 via nixpkgs commit 8110df5 (March 2026).
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";

    # Pinned iroh source for iroh-relay binary
    iroh-src = {
      url = "github:n0-computer/iroh/v0.91.2";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    flake-utils,
    iroh-src,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      # --- Native builds (host == target) ---
      craneLib = crane.mkLib pkgs;

      # Common source filtering for harmony workspace
      harmonySrc = pkgs.lib.fileset.toSource {
        root = ./.;
        # Only include Cargo manifests + Rust source. Avoids
        # fileset.toml which picks up non-Cargo TOML files and
        # widens the cache invalidation boundary.
        fileset = pkgs.lib.fileset.unions [
          (craneLib.fileset.cargoTomlAndLock ./.)
          (craneLib.fileset.rust ./.)
          # Include .cargo/config.toml so cross-build rustflags
          # (e.g., link-self-contained=yes for aarch64 musl) are visible.
          (pkgs.lib.fileset.maybeMissing ./.cargo/config.toml)
          # Include .wat files needed by build.rs (compiled to WASM at build time).
          ./crates/harmony-node/src/inference_runner.wat
        ];
      };

      harmonyCommonArgs = {
        src = harmonySrc;
        strictDeps = true;
        doCheck = false;
        pname = "harmony-node";
        version = "0.1.0";

        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
        ];
        buildInputs = with pkgs;
          [
            openssl
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            # darwin.apple_sdk.frameworks was removed in nixpkgs 26.05.
            # Security and SystemConfiguration are propagated transitively
            # by openssl — no explicit listing needed.
            pkgs.libiconv
          ];

        cargoExtraArgs = "-p harmony-node";
      };

      harmonyCargoArtifacts = craneLib.buildDepsOnly harmonyCommonArgs;

      harmony-node = craneLib.buildPackage (harmonyCommonArgs
        // {
          cargoArtifacts = harmonyCargoArtifacts;
        });

      # iroh-relay from pinned upstream source
      irohSrc = craneLib.cleanCargoSource iroh-src;

      irohCommonArgs = {
        src = irohSrc;
        strictDeps = true;
        doCheck = false;
        pname = "iroh-relay";
        version = "0.91.2";

        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
        ];
        buildInputs = with pkgs;
          [
            openssl
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];

        cargoExtraArgs = "-p iroh-relay --features server --bin iroh-relay";
      };

      irohCargoArtifacts = craneLib.buildDepsOnly irohCommonArgs;

      # Native build — dynamically links glibc/openssl. For local
      # development and testing only. For VM deployment, use the
      # cross-compiled static musl packages below:
      #   iroh-relay-x86_64-linux   (e2-micro GCP VMs)
      #   iroh-relay-aarch64-linux  (ARM VMs)
      iroh-relay = craneLib.buildPackage (irohCommonArgs
        // {
          cargoArtifacts = irohCargoArtifacts;
        });

      # --- Cross-compilation helper ---
      # Build a cross-compiled package given crossPkgs and a target triple.
      mkCross = {
        crossPkgs,
        cargoTarget,
      }: let
        crossCraneLib = crane.mkLib crossPkgs;

        # Common cross-compilation environment
        crossEnv = {
          strictDeps = true;
          doCheck = false;
          CARGO_BUILD_TARGET = cargoTarget;

          # Tell cargo/cc which linker and C compiler to use for the target
          "CARGO_TARGET_${builtins.replaceStrings ["-"] ["_"] (pkgs.lib.toUpper cargoTarget)}_LINKER" =
            "${crossPkgs.stdenv.cc}/bin/${crossPkgs.stdenv.cc.targetPrefix}cc";
          "CC_${builtins.replaceStrings ["-"] ["_"] cargoTarget}" =
            "${crossPkgs.stdenv.cc}/bin/${crossPkgs.stdenv.cc.targetPrefix}cc";

          # Ensure fully static binaries for musl targets. harmony-node
          # gets this from .cargo/config.toml, but iroh-relay uses its own
          # source tree, so set it explicitly for both.
          "CARGO_TARGET_${builtins.replaceStrings ["-"] ["_"] (pkgs.lib.toUpper cargoTarget)}_RUSTFLAGS" =
            "-C link-self-contained=yes";

          nativeBuildInputs = with crossPkgs; [
            buildPackages.pkg-config
            buildPackages.cmake
          ];
          buildInputs = with crossPkgs; [
            openssl
          ];

          # Tell pkg-config where to find cross-compiled libraries
          PKG_CONFIG_PATH = "${crossPkgs.openssl.dev}/lib/pkgconfig";
        };

        harmonyCrossArgs =
          crossEnv
          // {
            src = harmonySrc;
            pname = "harmony-node";
            version = "0.1.0";
            cargoExtraArgs = "-p harmony-node";
          };

        harmonyCrossArtifacts = crossCraneLib.buildDepsOnly harmonyCrossArgs;

        harmonyCross = crossCraneLib.buildPackage (harmonyCrossArgs
          // {
            cargoArtifacts = harmonyCrossArtifacts;
          });

        irohCrossSrc = crossCraneLib.cleanCargoSource iroh-src;

        irohCrossArgs =
          crossEnv
          // {
            src = irohCrossSrc;
            pname = "iroh-relay";
            version = "0.91.2";
            cargoExtraArgs = "-p iroh-relay --features server --bin iroh-relay";
          };

        irohCrossArtifacts = crossCraneLib.buildDepsOnly irohCrossArgs;

        irohCross = crossCraneLib.buildPackage (irohCrossArgs
          // {
            cargoArtifacts = irohCrossArtifacts;
          });
      in {
        harmony-node = harmonyCross;
        iroh-relay = irohCross;
      };

      # Cross-compiled variants (musl for static linking, avoids glibc cross issues)
      crossX86_64 = mkCross {
        crossPkgs = pkgs.pkgsCross.musl64;
        cargoTarget = "x86_64-unknown-linux-musl";
      };

      crossAarch64 = mkCross {
        crossPkgs = pkgs.pkgsCross.aarch64-multiplatform-musl;
        cargoTarget = "aarch64-unknown-linux-musl";
      };
    in {
      packages = {
        inherit harmony-node iroh-relay;
        default = harmony-node;

        # Cross-compiled Linux binaries
        harmony-node-x86_64-linux = crossX86_64.harmony-node;
        harmony-node-aarch64-linux = crossAarch64.harmony-node;
        iroh-relay-x86_64-linux = crossX86_64.iroh-relay;
        iroh-relay-aarch64-linux = crossAarch64.iroh-relay;
      };

      # Checks: nix flake check exercises native builds
      checks = {
        inherit harmony-node iroh-relay;
      };

      devShells.default = craneLib.devShell {
        packages = with pkgs; [
          pkg-config
          cmake
          openssl
          sccache
        ];

        # sccache: shared Rust compilation cache via Cloudflare R2.
        # Credentials live in ~/.config/sccache/config (see docs/sccache-setup.md).
        # CARGO_INCREMENTAL=0 is required — incremental builds are incompatible
        # with sccache (each machine's incremental artifacts are host-specific).
        RUSTC_WRAPPER = "${pkgs.sccache}/bin/sccache";
        CARGO_INCREMENTAL = "0";
      };
    });
}

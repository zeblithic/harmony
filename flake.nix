{
  description = "Harmony — decentralized internet stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
        fileset = pkgs.lib.fileset.unions [
          (craneLib.fileset.cargoTomlAndLock ./.)
          (craneLib.fileset.rust ./.)
          (craneLib.fileset.toml ./.)
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
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
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
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
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

      devShells.default = craneLib.devShell {
        packages = with pkgs; [
          pkg-config
          cmake
          openssl
        ];
      };
    });
}

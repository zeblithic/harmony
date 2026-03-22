{
  description = "Harmony — decentralized internet stack";

  inputs = {
    # nixos-unstable for latest musl/Rust toolchain. Consider pinning to
    # a stable release (e.g. nixos-24.11) once cross-compilation is validated.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";

    # Pinned iroh source for iroh-relay binary
    iroh-src = {
      url = "github:n0-computer/iroh/v0.91.2";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, crane, flake-utils, iroh-src }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # --- Native builds (host == target) ---
        craneLib = crane.mkLib pkgs;

        # Common source filtering
        harmonySrc = craneLib.cleanCargoSource ./.;

        # Common native build inputs
        commonNativeBuildInputs = with pkgs; [
          pkg-config
        ];
        commonBuildInputs = with pkgs; [
          openssl
        ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
          pkgs.apple-sdk_15
          pkgs.libiconv
        ];

        # harmony-node (native)
        harmonyCommonArgs = {
          src = harmonySrc;
          pname = "harmony-node";
          version = "0.1.0";
          cargoExtraArgs = "-p harmony-node";
          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs = commonBuildInputs;
        };

        harmonyDeps = craneLib.buildDepsOnly harmonyCommonArgs;

        harmony-node = craneLib.buildPackage (harmonyCommonArgs // {
          cargoArtifacts = harmonyDeps;
        });

        # iroh-relay (native)
        irohSrcCleaned = craneLib.cleanCargoSource iroh-src;

        irohRelayCommonArgs = {
          src = irohSrcCleaned;
          pname = "iroh-relay";
          version = "0.91.2";
          cargoExtraArgs = "-p iroh-relay --features server";
          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs = commonBuildInputs;
        };

        irohRelayDeps = craneLib.buildDepsOnly irohRelayCommonArgs;

        iroh-relay = craneLib.buildPackage (irohRelayCommonArgs // {
          cargoArtifacts = irohRelayDeps;
          cargoExtraArgs = "-p iroh-relay --features server --bin iroh-relay";
        });

        # --- Cross-compilation helpers ---
        # Only define cross targets when building on macOS or Linux
        mkCrossPackages = crossSystem:
          let
            crossPkgs = import nixpkgs {
              inherit system;
              crossSystem = {
                config = crossSystem;
              };
            };
            crossCraneLib = crane.mkLib crossPkgs;
            crossSrc = crossCraneLib.cleanCargoSource ./.;
            crossIrohSrc = crossCraneLib.cleanCargoSource iroh-src;

            crossOpenssl = crossPkgs.openssl;

            crossNativeBuildInputs = [
              crossPkgs.stdenv.cc
              crossPkgs.pkg-config
            ];
            crossBuildInputs = [
              crossOpenssl
            ];

            # Override .cargo/config.toml linker settings — Nix provides
            # the correct cross-linker via crossPkgs.stdenv.cc. The env var
            # CARGO_TARGET_<TRIPLE>_LINKER takes precedence over config.toml.
            targetUpper = builtins.replaceStrings ["-"] ["_"]
              (pkgs.lib.toUpper crossPkgs.stdenv.hostPlatform.rust.rustcTargetSpec);
            linkerEnv = {
              "CARGO_TARGET_${targetUpper}_LINKER" = "${crossPkgs.stdenv.cc}/bin/${crossPkgs.stdenv.cc.targetPrefix}cc";
            };

            crossHarmonyCommonArgs = {
              src = crossSrc;
              pname = "harmony-node";
              version = "0.1.0";
              cargoExtraArgs = "-p harmony-node";
              nativeBuildInputs = crossNativeBuildInputs;
              buildInputs = crossBuildInputs;
              CARGO_BUILD_TARGET = crossPkgs.stdenv.hostPlatform.rust.rustcTargetSpec;
              CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static";
              HOST_CC = "${pkgs.stdenv.cc}/bin/cc";
            } // linkerEnv;

            crossHarmonyDeps = crossCraneLib.buildDepsOnly crossHarmonyCommonArgs;

            crossHarmonyNode = crossCraneLib.buildPackage (crossHarmonyCommonArgs // {
              cargoArtifacts = crossHarmonyDeps;
            });

            crossIrohRelayCommonArgs = {
              src = crossIrohSrc;
              pname = "iroh-relay";
              version = "0.91.2";
              cargoExtraArgs = "-p iroh-relay --features server";
              nativeBuildInputs = crossNativeBuildInputs;
              buildInputs = crossBuildInputs;
              CARGO_BUILD_TARGET = crossPkgs.stdenv.hostPlatform.rust.rustcTargetSpec;
              CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static";
              HOST_CC = "${pkgs.stdenv.cc}/bin/cc";
            } // linkerEnv;

            crossIrohRelayDeps = crossCraneLib.buildDepsOnly crossIrohRelayCommonArgs;

            crossIrohRelay = crossCraneLib.buildPackage (crossIrohRelayCommonArgs // {
              cargoArtifacts = crossIrohRelayDeps;
              cargoExtraArgs = "-p iroh-relay --features server --bin iroh-relay";
            });
          in {
            harmony-node = crossHarmonyNode;
            iroh-relay = crossIrohRelay;
          };

        # Cross-compiled packages for Linux targets
        x86_64LinuxMusl = mkCrossPackages "x86_64-unknown-linux-musl";
        aarch64LinuxMusl = mkCrossPackages "aarch64-unknown-linux-musl";

      in {
        packages = {
          # Native builds
          inherit harmony-node iroh-relay;
          default = harmony-node;

          # Cross-compiled Linux builds
          harmony-node-x86_64-linux = x86_64LinuxMusl.harmony-node;
          harmony-node-aarch64-linux = aarch64LinuxMusl.harmony-node;
          iroh-relay-x86_64-linux = x86_64LinuxMusl.iroh-relay;
          iroh-relay-aarch64-linux = aarch64LinuxMusl.iroh-relay;
        };

        # Checks: nix flake check exercises native builds
        checks = {
          inherit harmony-node iroh-relay;
        };
      }
    );
}

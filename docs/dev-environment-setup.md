# Development Environment Setup

Complete guide to setting up a development machine for working on
Zeblithic Harmony repositories. Written from a fresh WSL/Debian install
(2026-04-01, Opal machine), but applicable as a checklist for any
Linux or macOS system.

---

## Prerequisites

What you need before anything else (may already be present on macOS
or established Linux installs):

```bash
# Debian/Ubuntu/WSL
sudo apt update
sudo apt install -y curl git

# macOS: install Xcode command line tools
xcode-select --install
```

---

## 1. GitHub Authentication

### Generate an SSH key

```bash
ssh-keygen -t ed25519 -C "your-machine-name" -f ~/.ssh/id_ed25519 -N ""
```

### Install GitHub CLI and authenticate

**Debian/Ubuntu/WSL:**

```bash
sudo apt install -y wget
sudo mkdir -p -m 755 /etc/apt/keyrings
wget -nv -O /tmp/gh-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg
sudo cp /tmp/gh-keyring.gpg /etc/apt/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli-stable.list
sudo apt update
sudo apt install -y gh
```

**macOS:**

```bash
brew install gh
```

**Authenticate (all platforms):**

```bash
gh auth login -p ssh -h github.com -w
```

This opens a browser for OAuth, uploads your SSH key, and configures
git to use SSH.

### Add GitHub's host key

```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
```

### Set git identity

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 2. Clone Repositories

We use the structure `~/work/<org-or-user>/<repo-name>`:

```bash
mkdir -p ~/work/zeblithic ~/work/steveyegge

# Core Harmony repos
for repo in harmony harmony-client harmony-os harmony-openwrt harmony-glitch; do
  gh repo clone "zeblithic/$repo" ~/work/zeblithic/$repo
done

# Beads issue tracker
gh repo clone steveyegge/beads ~/work/steveyegge/beads
```

Optional repos (not required for day-to-day development):

```bash
for repo in harmony-arch harmony-browser harmony-stq8; do
  gh repo clone "zeblithic/$repo" ~/work/zeblithic/$repo
done
```

---

## 3. Core Build Tools

### Debian/Ubuntu/WSL

```bash
sudo apt install -y \
  build-essential \
  pkg-config \
  cmake \
  libssl-dev \
  python3 \
  python3-pip
```

### macOS

```bash
# Xcode command line tools (from step 0) provide clang, make, etc.
brew install pkg-config cmake openssl python3
```

---

## 4. Rust Toolchain

Required by: **harmony** (core), **harmony-client**, **harmony-os**,
**harmony-glitch**, **harmony-openwrt**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

Verify: `rustc --version` (need >= 1.85)

### Additional Rust targets

For harmony-os bare-metal and harmony-openwrt cross-compilation:

```bash
# Musl targets (static linking for openwrt and OS)
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl

# Nightly toolchain (harmony-os bootloader crate)
rustup toolchain install nightly
```

**Linux only** — musl C toolchain for cross-compiling native dependencies:

```bash
sudo apt install -y musl-tools
```

---

## 5. Node.js and npm

Required by: **harmony-client**, **harmony-glitch** (Svelte/Vite/Tauri
frontends)

### Debian/Ubuntu/WSL (Node 22 LTS via NodeSource)

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

### macOS

```bash
brew install node@22
```

Verify: `node --version` (need >= 18), `npm --version`

---

## 6. Tauri v2 System Dependencies

Required by: **harmony-client**, **harmony-glitch** (desktop apps)

### Debian/Ubuntu/WSL

```bash
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  libjavascriptcoregtk-4.1-dev \
  libsoup-3.0-dev \
  libgtk-3-dev \
  librsvg2-dev \
  libglib2.0-dev \
  patchelf
```

### macOS

No extra packages needed — Tauri uses WebKit which ships with macOS.

---

## 7. Go Toolchain

Required by: **beads** (bd issue tracker). Needs Go >= 1.25 with CGO.

### Debian/Ubuntu/WSL

Debian repos won't have Go 1.25+. Install from the official tarball:

```bash
curl -fsSL https://go.dev/dl/go1.25.8.linux-amd64.tar.gz \
  | sudo tar -C /usr/local -xzf -
echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' >> ~/.bashrc
source ~/.bashrc
```

### macOS

```bash
brew install go
```

Verify: `go version` (need >= 1.25)

---

## 8. Nix Package Manager

Required by: **harmony** (core, dev shell), **harmony-os** (flake builds,
NixOS configs)

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

After installation, restart your shell, then enable flakes:

```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

Verify: `nix --version` and `nix flake --help`

### Cachix (Nix binary cache)

Avoids rebuilding what other machines have already built:

```bash
nix profile install nixpkgs#cachix
```

The zeblithic cache is already configured as a substituter in the
harmony-os flake. To **push** build artifacts (so other machines can
reuse them):

```bash
cachix authtoken <YOUR_TOKEN>
# After a nix build:
nix build .#<package> | cachix push zeblithic
```

---

## 9. `just` (Command Runner)

Required by: **harmony-os** (justfile build recipes)

```bash
cargo install just
```

Verify: `just --version`

---

## 10. harmony-os Extras

For bare-metal kernel testing and cross-compilation:

### Debian/Ubuntu/WSL

```bash
sudo apt install -y qemu-system-x86 qemu-system-arm mtools
```

### macOS

```bash
brew install qemu mtools
```

---

## 11. Build and Install beads (bd)

```bash
cd ~/work/steveyegge/beads
make install
```

This builds with CGO (required for embedded Dolt) and installs `bd` to
`~/.local/bin/`. Make sure `~/.local/bin` is on your PATH.

Verify: `bd version` (need >= 0.62.0)

---

## 12. Bootstrap beads in All Repos

Each harmony repo has an existing beads database on its GitHub remote.
Use `bd bootstrap` to pull it, then create the local config files that
bootstrap doesn't generate.

**Repeat for each repo:**

```bash
cd ~/work/zeblithic/harmony    # (or harmony-client, harmony-os, etc.)
bd bootstrap

# Fix permissions
chmod 700 .beads

# Create metadata.json — bootstrap doesn't generate this
cat > .beads/metadata.json << 'EOF'
{
  "database": "dolt",
  "backend": "dolt",
  "dolt_mode": "embedded",
  "dolt_database": "beads"
}
EOF

# Create lock file and interactions log
touch .beads/embeddeddolt/.lock
touch .beads/interactions.jsonl

# Create config.yaml with auto-sync
cat > .beads/config.yaml << 'EOF'
backup:
  enabled: false

dolt.auto-push: true
dolt.auto-commit: "on"
EOF

# Verify
bd ready
```

> **Why the manual steps?** `bd bootstrap` pulls the Dolt database from
> the remote into `.beads/embeddeddolt/beads/` but does not create
> `metadata.json` (which tells bd the database name), `config.yaml`,
> or the `.lock` file. Without `metadata.json`, every bd command fails
> with `"no database selected"`. These files are gitignored and must be
> recreated on each new machine.

See also: [beads-dolt-setup.md](beads-dolt-setup.md) for detailed
Dolt operations, multi-machine sync, and troubleshooting.

---

## Verification Checklist

After completing all steps, verify everything builds:

```bash
# Harmony core (40 Rust crates)
cd ~/work/zeblithic/harmony && cargo check --workspace

# harmony-client (Tauri desktop app)
cd ~/work/zeblithic/harmony-client
npm install && npm run build                    # Frontend
cd src-tauri && cargo check                     # Rust backend

# harmony-glitch (Tauri game)
cd ~/work/zeblithic/harmony-glitch
npm install && npm run build                    # Frontend
cd src-tauri && cargo check                     # Rust backend

# harmony-os (bare-metal OS)
cd ~/work/zeblithic/harmony-os && cargo check --workspace

# harmony-openwrt (musl cross-compile check)
cd ~/work/zeblithic/harmony
cargo check -p harmony-node --target x86_64-unknown-linux-musl

# beads
bd version
cd ~/work/zeblithic/harmony && bd ready
```

---

## Toolchain Summary

| Tool | Version | Used By |
|------|---------|---------|
| Rust (stable) | >= 1.85 | harmony, harmony-client, harmony-os, harmony-glitch, harmony-openwrt |
| Rust (nightly) | latest | harmony-os (bootloader crate) |
| Node.js | >= 18 (22 LTS recommended) | harmony-client, harmony-glitch |
| Go | >= 1.25 (with CGO) | beads |
| Nix | >= 2.18 (with flakes) | harmony (dev shell), harmony-os (builds) |
| just | any | harmony-os |
| QEMU | any | harmony-os (testing) |
| cachix | any | Nix binary cache sharing |
| gh | any | GitHub CLI |
| bd | >= 0.62.0 | Issue tracking across all repos |

### System packages (Debian/Ubuntu/WSL)

```
build-essential pkg-config cmake libssl-dev python3
libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev libsoup-3.0-dev
libgtk-3-dev librsvg2-dev libglib2.0-dev patchelf
musl-tools qemu-system-x86 qemu-system-arm mtools
```

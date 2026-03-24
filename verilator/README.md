# 🧹 Verilator Uninstall Cheatsheet

## 🔍 0. Find what you’re using

```bash
which verilator
verilator --version
```

Example (yours):

```bash
/home/eric891224/.usr/bin/verilator
```

---

## 🧠 1. Identify install type

### Case A — User-local install (your case)

Prefix like:

```bash
~/.usr
~/.local
~/opt/verilator
```

### Case B — System package

```bash
/usr/bin/verilator
```

### Case C — Built from source (still have repo)

---

## 🗑️ 2. Uninstall commands

---

## ✅ Case A — User-local install (🔥 YOUR CASE)

Remove everything under the prefix:

```bash
rm -f ~/.usr/bin/verilator*
rm -rf ~/.usr/share/verilator
rm -f ~/.usr/share/man/man1/verilator*
```

Optional (clean includes if present):

```bash
rm -rf ~/.usr/include/verilator
```

Then refresh shell cache:

```bash
hash -r
```

Verify:

```bash
which verilator
```

---

## ✅ Case B — Installed via package manager

### Ubuntu / Debian

```bash
sudo apt remove verilator
```

### Fedora / CentOS

```bash
sudo dnf remove verilator
```

### Arch

```bash
sudo pacman -R verilator
```

---

## ✅ Case C — Installed from source (repo still exists)

Go to the build directory:

```bash
cd verilator
make uninstall
```

⚠️ If this fails → fallback to manual delete (Case A style)

---

## 🧪 3. Double-check no leftovers

```bash
which -a verilator
```

If multiple paths show up → remove all stale ones.

Also check:

```bash
echo $PATH | tr ':' '\n' | grep usr
```

---

## ⚠️ 4. Common gotchas

### ❌ Still seeing old version?

You probably have:

- multiple installs
- PATH priority issue

Fix:

```bash
hash -r
```

---

### ❌ `VERILATOR_ROOT` set

```bash
echo $VERILATOR_ROOT
```

If set incorrectly:

```bash
unset VERILATOR_ROOT
```

---

## 🚀 5. Reinstall clean (recommended)

```bash
git clone https://github.com/verilator/verilator
cd verilator
autoconf
./configure --prefix=$HOME/.usr
make -j$(nproc)
make install
```

---

## 🧾 TL;DR (your exact fix)

Just run:

```bash
rm -f ~/.usr/bin/verilator*
rm -rf ~/.usr/share/verilator
rm -f ~/.usr/share/man/man1/verilator*
hash -r
which verilator
```

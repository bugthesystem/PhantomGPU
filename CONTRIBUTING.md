# Contributing to PhantomGPU

We welcome contributions to PhantomGPU! This guide helps you get started.

## Getting Started

### Prerequisites
- Rust 1.75+
- Python 3.8+ (for TensorFlow support)
- Git

### Development Setup
```bash
# Clone the repository
git clone https://github.com/bugthesystem/phantom-gpu.git
cd phantom-gpu

# Install dependencies
cargo build --features real-models

# Run tests
cargo test

# Install development tools
cargo install cargo-watch
cargo install cargo-clippy
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write clean, documented code
- Add tests for new functionality
- Follow Rust conventions and `cargo clippy` suggestions

### 3. Test Your Changes
```bash
# Run all tests
cargo test

# Test with real models (if applicable)
cargo test --features real-models

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --features real-models
```

### 4. Submit a Pull Request
- Write a clear description of your changes
- Reference any related issues
- Ensure CI tests pass

## Types of Contributions

### 🐛 Bug Reports
- Use the issue template
- Include minimal reproduction steps
- Provide system information (OS, Rust version, etc.)

### ✨ Feature Requests
- Check existing issues first
- Describe the problem you're solving
- Provide examples of the expected behavior

### 🔧 Code Contributions
**High Priority Areas:**
- Web interface development (WASM/TypeScript)
- Additional ML framework support
- Cloud provider integrations
- Performance optimizations

**Good First Issues:**
- Documentation improvements
- Error message enhancements
- Additional test coverage
- Example model scripts

### 📚 Documentation
- Update README for new features
- Add examples for common use cases
- Improve API documentation
- Create tutorials

## Code Guidelines

### Rust Code Style
- Follow `rustfmt` formatting
- Use `cargo clippy` and fix warnings
- Write documentation comments for public APIs
- Add unit tests for new functions

### Error Handling
- Use `PhantomGpuError` for domain errors
- Provide helpful error messages
- Include suggestions for fixing issues

### Performance
- Avoid unnecessary allocations
- Use async/await for I/O operations
- Profile performance-critical code

## Testing

### Unit Tests
```bash
# Run specific module tests
cargo test emulator

# Run with output
cargo test -- --nocapture
```

### Integration Tests
```bash
# Test real model loading
cargo test --features real-models test_model_loading

# Test CLI commands
cargo test --features real-models cli_tests
```

### Hardware Profile Testing
```bash
# Test custom hardware profiles
cargo test --features real-models hardware_profile_tests
```

## Architecture

### Core Components
- `src/emulator.rs` - GPU emulation engine
- `src/real_model_loader.rs` - Model loading and analysis
- `src/tensorflow_parser.rs` - TensorFlow-specific parsing
- `src/real_hardware_model.rs` - Hardware performance modeling
- `src/cli.rs` - Command-line interface

### Adding New Features

#### New ML Framework Support
1. Add parsing logic in a new `src/{framework}_parser.rs`
2. Update `ModelFormat` enum in `real_model_loader.rs`
3. Add CLI support in `cli.rs`
4. Write comprehensive tests

#### New GPU Support
1. Add GPU definition to `hardware_profiles.toml`
2. Include thermal, memory, and compute characteristics
3. Test with various model types
4. Update documentation

#### New CLI Commands
1. Add command to `Commands` enum in `cli.rs`
2. Implement handler in `commands.rs`
3. Add tests for the new command
4. Update help documentation

## Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist
- [ ] Update version in `Cargo.toml`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Test with real models
- [ ] Update documentation
- [ ] Create release on GitHub

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an issue with the bug template
- **Security**: Email security@phantomgpu.dev (if applicable)
- **Chat**: Join our Discord server (if applicable)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Focus on technical discussions
- Help newcomers learn and contribute
- Report inappropriate behavior

## License

By contributing to PhantomGPU, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PhantomGPU! 🚀 
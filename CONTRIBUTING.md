# Contributing Guidelines

Thank you for your interest in contributing to the Awkward + MonetDB Hybrid HEP Analysis Engine.
This project is experimental and research-oriented, and contributions are welcome in all areas.

---

## How to Contribute

### 1. File Issues
Use GitHub Issues to report:

- Bugs
- Performance regressions
- Feature requests
- Questions about architecture or design

Please include:

- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, MonetDB version, OS)

---

### 2. Submit Pull Requests

#### PR Requirements
- Clear description of the change
- Reference to related issues
- Tests for new functionality
- Benchmarks if performance-related
- Documentation updates when relevant

#### PR Process
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request
5. Participate in review

---

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Prefer functional, vectorized operations (Awkward, NumPy)
- Avoid Python loops over events or particles

### SQL
- Keep queries simple and event-level
- Avoid UNNEST; nested logic belongs in Awkward
- Use indexes on `event_id`

---

## Areas Where Help Is Needed

### Ingestion
- Parallel ROOT → Awkward → Arrow → MonetDB pipeline
- Support for multiple particle types
- Compression and batching strategies

### Analysis
- Implement ADL Q1–Q8
- Add physics helper functions
- Benchmark against ROOT/RDataFrame

### Infrastructure
- Docker environment
- CI for ingestion + analysis tests
- Benchmark suite

### Documentation
- Tutorials
- Architecture diagrams
- Example notebooks

---

## Communication
Use GitHub Discussions or Issues for coordination.
Large design changes should be proposed before implementation.

---

## License
By contributing, you agree that your contributions will be licensed under the project's license (MIT or BSD-3).

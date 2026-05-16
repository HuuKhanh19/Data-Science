# Changelog

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Semantic versioning: x.y.z (x = major methodology change, y = feature, z = fix).

## [Unreleased]

## [0.1.0] - 2026-05-16

### Added
- Initial repo bootstrap.
- Folder structure theo `IMPLEMENTATION.md` section 3.
- Utility modules: `config.py` (frozen `CFG` dataclass), `seeds.py` (global seed setup), `logging.py` (project-wide logger format).
- Project documentation: `README.md`, `LICENSE` (MIT), `pyproject.toml`, `requirements.txt`, `.gitignore`.
- Placeholder structure cho `src/{data,features,models,eval,interpret,ablation,storage,app,inference,utils}/`, `tests/`, `notebooks/`, `scripts/`.

### Notes
- Chưa lock pre-registration. Lock dự kiến cuối tuần 2 (28/05/2026) với git tag `v1.0-prereg-locked`.
- `research_design.md` và `IMPLEMENTATION.md` được upload lên project knowledge trên Claude, sẽ commit vào repo trong Session 2 (sau khi resolve Open Q1 về adjusted close).

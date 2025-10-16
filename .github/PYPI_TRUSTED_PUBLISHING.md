# PyPI Trusted Publishing Setup

This repository uses PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) feature for secure, token-less releases. Trusted Publishing uses OpenID Connect (OIDC) to verify that GitHub Actions is authorized to publish packages to PyPI.

## Benefits

- **No API tokens to manage**: No need to create, rotate, or secure PyPI API tokens
- **Better security**: Short-lived credentials that can't be leaked
- **Simpler setup**: No secrets to configure in GitHub
- **Audit trail**: Clear connection between releases and GitHub Actions runs

## Initial Setup

### 1. Configure PyPI Trusted Publisher

1. Go to your PyPI project: https://pypi.org/project/track-linearization/
2. Navigate to **Manage** → **Publishing**
3. Click **Add a new publisher**
4. Fill in the form:
   - **PyPI Project Name**: `track-linearization`
   - **Owner**: `LorenFrankLab`
   - **Repository name**: `track_linearization`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi` (optional but recommended)

5. Click **Add**

### 2. Set up GitHub Environment (Recommended)

For additional security, create a GitHub environment:

1. Go to repository **Settings** → **Environments**
2. Click **New environment**
3. Name it `pypi`
4. Configure protection rules (optional):
   - **Required reviewers**: Add reviewers who must approve deployments
   - **Wait timer**: Add a delay before deployment
   - **Deployment branches**: Restrict to tags matching `v*`

### 3. Verify Workflow Configuration

The workflow in `.github/workflows/release.yml` is already configured with:

```yaml
publish:
  name: Publish to PyPI
  runs-on: ubuntu-latest
  needs: [test-package]
  if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
  environment:
    name: pypi
    url: https://pypi.org/project/track-linearization/
  permissions:
    id-token: write  # Required for trusted publishing
  steps:
    - name: Download distribution artifacts
      uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

Key points:
- `permissions.id-token: write` is required for OIDC
- `environment.name: pypi` matches the GitHub environment
- No `user` or `password` parameters needed!

## Usage

### Creating a Release

1. **Update version and changelog**:
   ```bash
   # Update CHANGELOG.md with new version section
   # Version is automatically determined from git tags via hatch-vcs
   ```

2. **Create and push a tag**:
   ```bash
   git tag v2.4.0
   git push origin v2.4.0
   ```

3. **Automated process**:
   - GitHub Actions runs the release workflow
   - Tests run across all Python versions
   - Package is built and validated
   - Package is uploaded to PyPI via Trusted Publishing
   - GitHub Release is created automatically

### Manual Testing Before Release

Test the build process without publishing:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the build
twine check dist/*

# Test installation locally
pip install dist/*.whl
```

## Troubleshooting

### "OIDC token verification failed"

- Verify the Trusted Publisher configuration on PyPI matches exactly:
  - Repository owner: `LorenFrankLab`
  - Repository name: `track_linearization`
  - Workflow filename: `release.yml`
  - Environment name: `pypi` (if configured)

### "Missing id-token permission"

- Ensure the workflow has:
  ```yaml
  permissions:
    id-token: write
  ```

### "Package already exists"

- You cannot overwrite existing versions on PyPI
- Create a new version tag to publish a new release

## Migration from API Tokens

If you previously used API tokens:

1. Complete the Trusted Publishing setup above
2. Remove the `PYPI_API_TOKEN` secret from GitHub repository settings
3. The workflow will automatically use Trusted Publishing instead

## Security Considerations

- **Environment protection**: Use GitHub environments with required reviewers for production releases
- **Branch protection**: Configure tag protection rules to prevent unauthorized tags
- **Workflow permissions**: The workflow only has `id-token: write` permission, minimizing attack surface
- **Audit logs**: All deployments are logged in both GitHub Actions and PyPI

## References

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPA Publishing Action](https://github.com/pypa/gh-action-pypi-publish)

## Support

For issues with Trusted Publishing:
- PyPI Help: https://pypi.org/help/
- GitHub Actions Discussions: https://github.com/orgs/community/discussions/categories/actions

# Documentation Deployment Setup

This file explains how to configure GitHub Pages to serve the pyLattice documentation.

## Automatic Deployment

The repository now includes a GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) that automatically builds and deploys the Sphinx documentation to GitHub Pages.

### How it works:

1. **Trigger**: The workflow runs on every push to the `master` or `main` branch
2. **Build**: It installs Python dependencies and builds the documentation using Sphinx
3. **Deploy**: It automatically deploys the built HTML to GitHub Pages

### GitHub Pages Configuration Required:

To enable this automatic deployment, you need to configure GitHub Pages in your repository settings:

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. Save the configuration

### First-time Setup:

After merging this PR and configuring the Pages settings:

1. The workflow will run automatically on the next push to `master`/`main`
2. Your documentation will be available at: `https://tcadart.github.io/pyLattice/`
3. Updates to the documentation will be automatically deployed when you push changes

### Manual Building (for development):

To build the documentation locally:

```bash
cd docs
pip install sphinx myst-parser sphinx-rtd-theme sphinx-autodoc-typehints
make html
```

The built documentation will be in `docs/build/html/`.

### Troubleshooting:

- **404 Error**: Make sure GitHub Pages is configured to use "GitHub Actions" as the source
- **Build Failures**: Check the Actions tab in your repository for build logs
- **Missing Dependencies**: The workflow automatically installs required Python packages

### Files Added/Modified:

- `.github/workflows/deploy-docs.yml`: GitHub Actions workflow for building and deploying docs
- `docs/source/conf.py`: Updated to automatically create `.nojekyll` file for GitHub Pages
- `docs/deploy-instructions.md`: This instruction file
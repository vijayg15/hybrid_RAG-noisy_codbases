from __future__ import annotations
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from git import Repo
from app.core.config import get_settings
from app.utils.hashing import sha256_text


class RepositoryService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _authenticated_url(repo_url: str, token: str | None) -> str:
        if not token or not repo_url.startswith(("https://", "http://")):
            return repo_url
        parts = urlsplit(repo_url)
        # Works for GitHub/GitLab/Bitbucket HTTPS token authentication.
        return urlunsplit((parts.scheme, f"oauth2:{token}@{parts.netloc}", parts.path, parts.query, parts.fragment))

    def clone_or_update(self, repo_url: str, token: str | None = None, branch: str | None = None) -> tuple[str, Path, str]:
        repo_id = sha256_text(re.sub(r"[^a-zA-Z0-9:/._-]", "", repo_url))[:16]
        target = self.settings.data_dir / "repos" / repo_id
        auth_url = self._authenticated_url(repo_url, token)
        if target.exists() and (target / ".git").exists():
            repo = Repo(target)
            origin = repo.remotes.origin
            origin.set_url(auth_url)
            origin.fetch(prune=True)
            if branch:
                repo.git.checkout(branch)
                repo.git.reset("--hard", f"origin/{branch}")
            else:
                repo.git.reset("--hard", f"origin/{repo.active_branch.name}")
        else:
            shutil.rmtree(target, ignore_errors=True)
            repo = Repo.clone_from(auth_url, target, branch=branch)
        commit_sha = repo.head.commit.hexsha
        # Never persist tokenized origin URL.
        repo.remotes.origin.set_url(repo_url)
        return repo_id, target, commit_sha

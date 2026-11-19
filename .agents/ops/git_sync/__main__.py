from lib.checkout_lock import run_cli

if __name__ == "__main__":
    run_cli("git_sync.cli", release_before_main=True)

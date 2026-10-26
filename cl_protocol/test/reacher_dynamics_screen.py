"""Compatibility CLI; implementation moved to run_reacher_experiments.py."""
if __package__:
    from .run_reacher_experiments import main
else:
    from run_reacher_experiments import main

if __name__ == '__main__':
    main()

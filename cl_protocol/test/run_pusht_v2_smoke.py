"""Compatibility CLI; implementation moved to run_pusht_experiments.py."""
if __package__:
    from .run_pusht_experiments import main
else:
    from run_pusht_experiments import main

if __name__ == '__main__':
    main()

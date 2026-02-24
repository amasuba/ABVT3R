#!/usr/bin/env python3
"""
Smoke test runner for the pipeline.

Runs `Integration.start(1)` using the included `data_collection/` files
and writes outputs to `reconstruction_output/`.

This does not require Kinect hardware but does require the Python
dependencies listed in `requirements.txt`.
"""
import sys
import traceback

try:
    from integration import Integration
    # The RF model was saved when random_forest_class.py ran as __main__,
    # so pickle encoded trees as __main__.DecisionTreeRegressor.
    # Register the classes in __main__ so np.load(allow_pickle=True) can find them.
    from classes.random_forest_class import DecisionTreeRegressor, RandomForestRegressor
    sys.modules['__main__'].DecisionTreeRegressor = DecisionTreeRegressor
    sys.modules['__main__'].RandomForestRegressor = RandomForestRegressor
except Exception as e:
    print("Failed to import Integration. Ensure you're running from the project root and dependencies are installed.")
    print(e)
    sys.exit(2)


def progress_cb(message, percentage=None):
    pct = f"{percentage}" if percentage is not None else "?"
    print(f"[{pct}%] {message}")


def main(plant_id=1):
    integ = Integration(progress_callback=progress_cb)
    try:
        integ.start(plant_id)
        print("Smoke run completed successfully.")
        return 0
    except Exception as e:
        print("Smoke run failed:")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

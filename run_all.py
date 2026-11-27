import time

import chart
import compare_and_stress
import distance_check
import fd_monte_carlo
import rates_and_summary
import validation

CHECK_DISTANCE_MINIMIZATION = False


def main():
    start = time.time()
    validation.main()
    results = fd_monte_carlo.run()
    rates_and_summary.main(results)
    chart.main(results)
    if CHECK_DISTANCE_MINIMIZATION:
        distance_check.main(results)
    compare_and_stress.main()
    print(f"\ntotal time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

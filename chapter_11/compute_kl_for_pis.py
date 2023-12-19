import math


if __name__ == "__main__":
    # Table 11.1
    PI = (0.5, 0.5)
    PI_NEW = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

    def compute_probability_ratio(pi_new, pi):
        return pi_new / pi

    def compute_kl(pi_new, pi):
        result = 0.0
        for new, old in zip(pi_new, pi):
            result += new * math.log(new / old)

        return result

    for pi_new in PI_NEW:
        ratio_a1 = compute_probability_ratio(pi_new[0], PI[0])
        ratio_a2 = compute_probability_ratio(pi_new[1], PI[1])
        kl = compute_kl(pi_new, PI)

        print(f'pi_new: {pi_new}, ratio_a1: {ratio_a1:.2f}, ratio_a2: {ratio_a2:.2f}, kl: {kl:.2f}')

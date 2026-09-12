import pickle
import random

import numpy as np

from generate_users2 import Simulated_User


def create_fixed_user_sets(
    num_train=1000,
    num_eval=125,
    num_test=125,
    seed=545,
):
    random.seed(seed)
    np.random.seed(seed)

    train_users = [Simulated_User("train") for _ in range(num_train)]
    eval_users = [Simulated_User("eval") for _ in range(num_eval)]
    test_users = [Simulated_User("test") for _ in range(num_test)]

    with open("data/user_train_set.pkl", "wb") as f:
        pickle.dump(train_users, f)

    with open("data/user_eval_set.pkl", "wb") as f:
        pickle.dump(eval_users, f)

    with open("data/user_test_set.pkl", "wb") as f:
        pickle.dump(test_users, f)

    print(f"Saved {num_train} train users to data/user_train_set.pkl")
    print(f"Saved {num_eval} eval users to data/user_eval_set.pkl")
    print(f"Saved {num_test} test users to data/user_test_set.pkl")


if __name__ == "__main__":
    create_fixed_user_sets()

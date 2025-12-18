def drow_policy(policy,env):
    arrow = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "·"}

    policy = policy.argmax(axis=1).reshape(env.size, env.size)

    for i in range(env.size):
        row = []
        for j in range(env.size):
            a = arrow[policy[i, j]]

            if env.state_id(i, j) in env.forbidden:
                cell = f"🪨{a}"
            elif env.state_id(i, j) in env.terminal:
                cell = f"🚩{a}"
            else:
                cell = f" {a} "

            row.append(cell)
        print(" ".join(row))
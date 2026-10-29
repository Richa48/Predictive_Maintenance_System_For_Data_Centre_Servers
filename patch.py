t = open("train.py").read()
a = 'choices=["rf", "xgb"]'
b = 'else:\n    clf = RandomForestClassifier('
if "hgb" in t:
    print("Already patched.")
elif a not in t or b not in t:
    print("ERROR: train.py is not the expected version.")
else:
    t = t.replace(a, 'choices=["rf", "xgb", "hgb"]')
    t = t.replace(b, 'elif args.algo == "hgb":\n    from sklearn.ensemble import HistGradientBoostingClassifier\n    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6,\n                                         class_weight="balanced", random_state=0)\n' + b)
    open("train.py", "w").write(t)
    print("Patched OK")

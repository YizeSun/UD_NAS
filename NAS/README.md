There are two different version of dqas added.
Both of them can be used for structure learning with differnt objective function, they will be updated later.

The ##matrix## version is marked with suffix "matrix" and takes distance between two matrixs as the objective function.
The other one uses expected value by measurement for the objective function.

1. How to run the code?

    1.1 Create you env with python3.10 (tested already) or other version if it fits for you.

    1.2 Install some packages for example [qiskit](https://qiskit.org/documentation/getting_started.html), [pennylane](https://pennylane.ai/install.html), [torch](https://pytorch.org/get-started/locally/) etc.

    1.3 Config your hyperparameters in file train_ud (e.g., marked arguments with "# *")

    1.4 Config your parameter "ops" as your operation pool which includes gates and their operation ranges.

    1.5 Set a name for your trainning task in the last three line (e.g., line 180 in file train_ud.py, with task name "'a0-rzcnotrzhcnot-ep5000-ls9-lr01")

    1.6 run 
        ```
            python train_ud.py or train_ud_matrix.py
        ```
    to start your training

2. waiting for continue(just wechat me for any questions)

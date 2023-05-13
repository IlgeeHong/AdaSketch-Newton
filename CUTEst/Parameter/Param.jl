AugLag = Parameter.AugLagParams(true,
                1e4,   # Max_Iter
                1e-4,  # EPS_Res
                1.0,   # mu
                0.1,   # beta
                1      # Rep
                )
Aug = Parameter.AugParams(true,
                1e4,     # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # eta1
                0.1,     # eta2
                0.1,     # delta
                0.1,     # xi_B
                0.1,     # beta
                1.5,     # nu
                1,       # Rep
                )
L1Adap = Parameter.L1AdapParams(true,
                1e4,     # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # penalty parameter
                0.1,     # kappa
                1e-8,    # eta
                0.1,     # xi_B
                1.5,     # nu
                1,       # Rep
                )
L1 = Parameter.L1Params(true,
                1e4,     # Max_Iter
                1e-4,    # EPS_Res
                1.0,     # penalty parameter
                1.0,     # kappa
                0.1,     # kappa1
                0.1,     # epsilon
                0.1,     # tau
                1e-8,    # eta
                0.1,     # xi_B
                1,       # Rep
                )

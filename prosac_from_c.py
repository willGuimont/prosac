def prosac(sorted_matches):
    # TODO: move this distCoeffs out!
    distCoeffs = np.zeros((5, 1))  # assume zero for now

    CORRESPONDENCES = sorted_matches.shape[0]
    isInlier = np.zeros([1,CORRESPONDENCES])
    SAMPLE_SIZE = 4
    MAX_OUTLIERS_PROPORTION = 0.8
    P_GOOD_SAMPLE = 0.99
    TEST_NB_OF_DRAWS = 60000
    TEST_INLIERS_RATIO = 0.5
    BETA = 0.01
    ETA0 = 0.05

    def niter_RANSAC(p, epsilon, s, Nmax):
        if(Nmax == -1):
            Nmax = np.iinfo(np.int32).max
        if(not (Nmax >= 1)):
            print("C++ Assertion failed - 1")
        if(epsilon <= 0):
            return 1
        logarg = - np.exp(s * np.log(1 - epsilon))
        logval = np.log(1 + logarg)
        N = np.log(1 - p) / logval
        if(logval < 0 and N < Nmax):
            return np.ceil(N)
        return Nmax

    def Imin(m, n, beta):
        mu = n*beta
        sigma = np.sqrt(n * beta * (1 - beta))
        return  np.ceil(m + mu + sigma * np.sqrt(2.706))

    def findSupport(n, isInlier):
        total_inliers = 0
        for i in range(0,n):
            total_inliers = total_inliers + isInlier[0,i]
        return total_inliers, isInlier

    N = CORRESPONDENCES
    m = SAMPLE_SIZE
    T_N = niter_RANSAC(P_GOOD_SAMPLE,MAX_OUTLIERS_PROPORTION,SAMPLE_SIZE,-1)
    beta = BETA
    I_N_min = (1 - MAX_OUTLIERS_PROPORTION)*N
    logeta0 = np.log(ETA0)

    # print("PROSAC sampling test\n")
    # print("number of correspondences (N):{0}\n".format(N))
    # print("sample size (m):{0}\n".format(m))
    # print("showing the first {0} draws from PROSAC\n".format(TEST_NB_OF_DRAWS))

    n_star = N
    I_n_star = 0
    I_N_best = 0
    t = 0
    n = m
    T_n = T_N

    for i in range(m):
        T_n = T_n * (n - i) / (N - i)

    T_n_prime = 1
    k_n_star = T_N

    best_model = {}
    while (((I_N_best < I_N_min) or t <= k_n_star) and t < T_N and t <= TEST_NB_OF_DRAWS):
        model = {}
        inliers = []
        t = t + 1
        # print("Iteration t=%d, " % t)

        if ((t > T_n_prime) and (n < n_star)):
            T_nplus1 = (T_n * (n+1)) / (n+1-m)
            n = n + 1
            T_n_prime = T_n_prime + np.ceil(T_nplus1 - T_n)
            T_n = T_nplus1

        if (t > T_n_prime):
            pts_idx = np.random.choice(n, m, replace=False)
        else:
            pts_idx = np.append(np.random.choice(n-1, m-1, replace=False), n-1)

        sample = sorted_matches[pts_idx]

        # 3. Model parameter estimation
        obj_points = sample[:, 2:5]
        img_points = sample[:, 0:2]

        img_points = np.ascontiguousarray(img_points[:, :2]).reshape((img_points.shape[0], 1, 2))  # this is required for SOLVEPNP_P3P
        retval, rvec, tvec = cv2.solvePnP(obj_points.astype(np.float32), img_points.astype(np.float32), K, distCoeffs, flags=cv2.SOLVEPNP_P3P)
        rotm = cv2.Rodrigues(rvec)[0]
        Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]
        model['Rt'] = Rt

        # 4. Model verification
        for i in range(len(sorted_matches)):  # run against all the other matches (all of them doesn't matter here)
            obj_point = sorted_matches[i, 2:5]
            img_point_gt = sorted_matches[i, 0:2]
            obj_point = np.r_[obj_point, 1]  # make homogeneous
            img_point_est = K.dot(Rt.dot(obj_point.transpose())[0:3])
            img_point_est = img_point_est / img_point_est[2]  # divide by last coordinate
            dist = np.linalg.norm(img_point_gt - img_point_est[0:2])
            if (dist < 8.0):
                isInlier[0,i] = 1
                inliers.append(sorted_matches[i])

        I_N, isInlier = findSupport(N, isInlier)
        # print("found {0} inliers!\n".format(I_N))

        if(I_N > I_N_best):
            I_N_best = I_N
            n_best = N
            I_n_best = I_N
            best_model = model

            if(1):
                epsilon_n_best = I_n_best / n_best
                I_n_test = I_N
                for n_test in range(N, m, -1):
                    if (not (n_test >= I_n_test)):
                        print("C++ Assertion failed - 3")
                    if ( (I_n_test * n_best > I_n_best * n_test) and (I_n_test > epsilon_n_best * n_test + np.sqrt(n_test * epsilon_n_best * (1 - epsilon_n_best) * 2.706) )):
                        if (I_n_test < Imin(m, n_test, beta)):
                            break
                        n_best = n_test
                        I_n_best = I_n_test
                        epsilon_n_best = I_n_best / n_best
                    I_n_test = I_n_test - isInlier[0, n_test - 1]

            if (I_n_best * n_star > I_n_star * n_best):
                if(not (n_best >= I_n_best)):
                    print("C++ Assertion failed - 2")
                n_star = n_best
                I_n_star = I_n_best
                k_n_star = niter_RANSAC(1 - ETA0, 1 - I_n_star / n_star, m, T_N)

    # print("PROSAC finished, reason:\n");
    # if(t > TEST_NB_OF_DRAWS):
    #     print("t={0} > max_t={1} (k_n_star={2}, T_N={3})\n".format(t, TEST_NB_OF_DRAWS, k_n_star, T_N))
    # elif(t > T_N):
    #     print("t={0} > T_N={1} (k_n_star={2})\n".format(t, T_N, k_n_star))
    # elif(t > k_n_star):
    #     print("t={0} > k_n_star={1} (T_N={2})\n".format(t ,k_n_star, T_N))

    inlier_no = I_N
    outliers_no = len(sorted_matches) - I_N
    iterations = t
    return inlier_no, outliers_no, iterations, best_model

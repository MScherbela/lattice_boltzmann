from lattice_boltzmann import LBMCalculator2D, run_animation_loop

nx = 400
ny = 100
tau = 0.6
n_t = 4000
delta_t = 50

lbm = LBMCalculator2D.build_from_file("data/test_text.png", [[(255,0,0), 0.1, 0.0]])
lbm.tau = 0.7
lbm.initialize()
run_animation_loop(lbm, n_t, delta_t)
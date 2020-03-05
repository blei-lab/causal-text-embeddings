from semi_parametric_estimation.att import att_estimates
import numpy as np
import os
import argparse
import pandas as pd

def main():
	outdir = os.path.join('..', 'out', args.data, args.experiment)
	for sim in os.listdir(outdir):
		mean_estimates = {'very_naive': [], 'q_only': [], 'plugin': [], 'one_step_tmle': [], 'aiptw': []}
		for split in os.listdir(os.path.join(outdir, sim)):
			if args.num_splits is not None:
				# print("ignoring split", split)
				if int(split) >= int(args.num_splits):
					continue
			array = np.load(os.path.join(outdir, sim, split, 'predictions.npz'))
			g = array['g']
			q0 = array['q0']
			q1 = array['q1']
			y = array['y']
			t = array['t']
			estimates = att_estimates(q0, q1, g, t, y, t.mean(), truncate_level=0.03)
			for est, att in estimates.items():
				mean_estimates[est].append(att)

		if args.data == 'reddit':
			sim = sim.replace('beta01.0.', '')
			options = sim.split('.0.')
			p2 = options[0].replace('beta1', '')
			p3 = options[1].replace('gamma', '')

			print("------ Simulation setting: Confounding strength =", p2, "; Variance:", p3, "------")
			print("True effect = 1.0")
		else:
			ground_truth_map = {'1.0':0.06, '5.0':0.06, '25.0':0.03}
			print("------ Simulation setting: Confounding strength =", sim)
			print("True effect = ", ground_truth_map[sim])


		for est, atts in mean_estimates.items():
			print('\t', est, np.round(np.mean(atts), 3), "+/-", np.round(np.std(atts),3))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", action="store", default="reddit")
	parser.add_argument("--experiment", action="store", default="base_model")
	parser.add_argument("--num-splits", action="store", default=None)
	args = parser.parse_args()

	main()
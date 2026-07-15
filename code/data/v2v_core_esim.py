"""ESIM-style video-to-events emulator (V2V core).

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  DATA PREP / training-data synthesis in
  data prep -> training -> member inference -> fusion -> submission.
  This is the event simulator behind all synthetic-event training data:
  syn_ev_data.SynEvDataset (EMA-E / VFIMamba-E finetuning) and
  scripts/gen_refiner_data.py (fusion-refiner training set) both call
  EventEmulator.video_to_voxel on temporally-upsampled grayscale clips.
  Its thresholds/noise are randomized by the callers within ranges calibrated
  to the real EventAid sensor (scripts/analyze_event_stats.py).

Model: an idealized DVS pixel integrates log-intensity change into a
per-pixel "potential"; every time the potential crosses +pos_thres
(-neg_thres) it emits one positive (negative) event and resets by one
threshold step, so a large change can emit several events per frame pair.

Input:  video, float array (N, H, W), grayscale intensities in [0, 255].
Output: video_to_voxel returns (N-1, H, W) signed integer event counts per
        interval (positive minus negative events at each pixel), which the
        callers rebin into n-bin temporal voxels.

Usage example:
  from evlib.v2v_core_esim import EventEmulator
  emu = EventEmulator(pos_thres=0.3, neg_thres=0.3, base_noise_std=0.01)
  slices = emu.video_to_voxel(gray_clip)  # gray_clip: (N, H, W) in [0, 255]
"""
import numpy as np

def reverse_gamma_correction(imgs, gamma=2.2):
	# Undo display gamma: map [0,255] sRGB-ish values back to linear light,
	# since the DVS model thresholds changes in *linear* log-intensity.
	return (imgs / 255) ** gamma * 255

class EventEmulator(object):
	"""Idealized DVS pixel-array simulator with optional noise sources.

	Parameters:
	  pos_thres / neg_thres  log-intensity contrast thresholds (can differ,
	                         emulating pos/neg sensitivity asymmetry)
	  base_noise_std         per-frame Gaussian shot noise (std, log domain)
	  hot_pixel_fraction     fraction of pixels given a persistent bias
	  hot_pixel_std          std of that persistent hot-pixel bias
	  put_noise_external     False: noise perturbs the integrator potential
	                         (quantized into events); True: noise is added as
	                         floats to the output voxel instead
	  seed                   unused placeholder (callers rely on global RNG)
	"""

	def __init__(
			self,
			pos_thres: float = 0.2,
			neg_thres: float = 0.2,
			base_noise_std: float = 0.1,
			hot_pixel_fraction: float = 0.001,
			hot_pixel_std: float = 0.1,
			put_noise_external: bool = False,
			seed: int = None,
	):
		self.pos_threshold = pos_thres
		self.neg_threshold = neg_thres
		self.base_noise_std = base_noise_std
		self.hot_pixel_fraction = hot_pixel_fraction
		self.hot_pixel_std = hot_pixel_std
		self.put_noise_external = put_noise_external
		self.seed = seed

	def video_to_voxel(self, video):
		"""Simulate events for an (N, H, W) clip -> (N-1, H, W) signed counts."""
		N, H, W = video.shape
		# Initialize the potential uniform random between -neg_thres and pos_thres
		# (random phase, so identical clips don't produce pixel-locked events).
		self.potential = np.random.rand(H, W) * (self.pos_threshold + self.neg_threshold) - self.neg_threshold

		all_voxels = []
		# Reverse gamma correction will make video more linear.
		video = reverse_gamma_correction(video)
		# Log-intensity with a small epsilon to keep log finite at black pixels.
		log_imgs = np.log(0.001 + video/255.0)

		# The hot noise persists for the entire video
		hot_pixel_mask = np.random.rand(H, W) < self.hot_pixel_fraction
		hot_noise = self.hot_pixel_std * np.random.randn(H, W)
		hot_noise = np.where(hot_pixel_mask, hot_noise, 0)		
		
		for i in range(N-1):
			# Integrate the log-intensity change of this frame interval.
			diff = log_imgs[i+1] - log_imgs[i]
			self.potential += diff
			base_noise = self.base_noise_std * np.random.randn(H, W)
			
			if not self.put_noise_external:
				# The noise influences the potential.
				self.potential += base_noise
				self.potential += hot_noise

			# Number of whole positive thresholds crossed at each pixel
			# (multiple events per interval possible for large changes).
			pos_events = np.floor_divide(self.potential, self.pos_threshold)
			pos_events = np.where(self.potential >= self.pos_threshold, pos_events, 0)
			
			# Same for negative threshold crossings.
			neg_events = np.floor_divide(-self.potential, self.neg_threshold)
			neg_events = np.where(self.potential <= -self.neg_threshold, neg_events, 0)

			# Reset: subtract the emitted charge, keep the sub-threshold residue.
			self.potential -= pos_events * self.pos_threshold
			self.potential += neg_events * self.neg_threshold

			# Signed per-pixel event count for this interval.
			voxel = pos_events - neg_events
			
			if self.put_noise_external:
				# Directly add the noise (a float) to the voxel output
				voxel = voxel + base_noise
				voxel = voxel + hot_noise

			all_voxels.append(voxel)
		
		return np.array(all_voxels)

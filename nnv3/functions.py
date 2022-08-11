import numpy as np

# TODO      ____\/__
# TODO    / .    .  \
# TODO   |   \/    v|  birb...
# TODO   \_________/
# TODO     |    |

# ?        ____\/__
# ?      / .    .  \
# ?     |   \/    v|  birb's friend...
# ?     \_________/
# ?       |    |


def split_batch(arr: np.ndarray, BATCH: int) -> np.ndarray:
	full: int = len(arr) // BATCH
	rem: int = len(arr) % BATCH

	output: list = []
	remainder: list =  []
	start: int = 0

	for i in range(full):
		end: int = (i + 1) * BATCH
		output.append(arr[start: end])
		start = end
	
	if rem > 0: remainder.append(arr[start:])

	return np.array(output), np.array(remainder)


def display_layer(_class: object, **kwargs) -> str:
	_class = type(_class).__name__
	# activation = type(activation).__name__
	initial: str = f"<{_class}"
	
	_keys: list = list(kwargs.keys())
	_vals: list = list(kwargs.values())

	for key, val in zip(_keys, _vals): 
		initial += f" {key}={val}"

	return f"{initial}>"


def uniform(shape: tuple) -> np.ndarray:
	return np.random.uniform(-0.5, 0.5, shape)

def normal(shape: tuple) -> np.ndarray:
	return np.random.normal(0, 1, shape)

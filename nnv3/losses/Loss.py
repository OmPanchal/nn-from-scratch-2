class Loss(object):
	def __init__(self, loss, loss_prime) -> None:
		self.loss = loss
		self.loss_prime = loss_prime

	def __str__(self) -> str:
		return type(self).__name__
# agent that takes in the target, any distractors and the vocabulary
# and sends a single word from the vocabulary
class AgnosticSender(Module):
  def __init__(self, input_dim=32, h_units=32):
      super(AgnosticSender, self).__init__()
      # sets the initial weights as values from a normal distribution
      w_init = torch.empty(input_dim, h_units).normal_(mean=0.0, std=0.01)
      # defines weights as a new tensor
      self.w = torch.nn.Parameter(torch.empty(input_dim, h_units)
      .normal_(mean=0.0, std=0.01), requires_grad=True)

      # sets the biases to contain zeroes
      b_init = torch.zeros(h_units)
      # defines biases as a new tensor
      self.b = torch.nn.Parameter(torch.zeros(h_units), requires_grad=True)
      # defines sigmoid function
      self.sig = Sigmoid()

  def forward(self, inputs):
      # returns the result of a Sigmoid function provided the inputs, weights
      # and the biases
      input = torch.mm(inputs, self.w).add(self.b)
      return self.sig(input)

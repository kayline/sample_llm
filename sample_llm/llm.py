import torch
import torch.nn.functional as F
# Read and format data
names  = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))
string_to_index = {s:i+1 for i,s in enumerate(chars)}
string_to_index['.'] = 0
index_to_string = {i:s for s,i in string_to_index.items()}
# Build bigrams
inputs, labels = [], []
for name in names:
	chars = ['.'] + list(name) + ['.']
	for inpt,label in zip(chars, chars[1:]):
		inpt_index  = string_to_index[inpt]
		label_index = string_to_index[label]
		inputs.append(inpt_index)
		labels.append(label_index)
inputs = torch.tensor(inputs)
num_inputs = inputs.nelement()
print(num_inputs)
labels = torch.tensor(labels)

# Build initial 'random' weights matrix
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)
print(inputs[23])
# Optimize
for k in range(50):
	# Optimize: generate some output
	inputs_enc = F.one_hot(inputs, num_classes=27).float()
	inputs_nums = inputs_enc @ W
	input_counts = inputs_nums.exp()
	input_probs = input_counts / input_counts.sum(1, keepdims=True)
	print('First row of probabilities', input_probs[0])
	loss = -input_probs[torch.arange(num_inputs), labels].log().mean() + + 0.01*(W**2).mean()
	print(loss.item())
	# Optimize: back
	W.grad = None
	loss.backward()
	# Optimize: adjust underlying weights
	W.data += -10 * W.grad
	print(W.exp()[0])

# Generate new names
for i in range(5):
	input_idx = 0
	gen_name = []

	while True:
		inpt_enc = F.one_hot(torch.tensor([input_idx]), num_classes=27).float()
		logits = inpt_enc @ W
		counts = logits.exp()
		p = counts / counts.sum(1, keepdims=True)

		inpt_idx = torch.multinomial(p, num_samples=1, replacement=True).item()
		gen_name.append(index_to_string[inpt_idx])
		if inpt_idx==0:
			break
	print(''.join(gen_name))





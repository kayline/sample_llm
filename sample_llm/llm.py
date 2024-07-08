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
labels = torch.tensor(labels)

# Build initial 'random' weights matrix
W = torch.randn((27,27), requires_grad=True)
# print(inputs[23])
# Optimize
loss = 0
for k in range(500):
	# Optimize: generate some output
	inputs_enc = F.one_hot(inputs, num_classes=27).float()
	inputs_nums = inputs_enc @ W
	input_counts = inputs_nums.exp()
	input_probs = input_counts / input_counts.sum(1, keepdims=True)
	loss = -input_probs[torch.arange(num_inputs), labels].log().mean() + + 0.01*(W**2).mean()
	# print("Loss: ", loss.item())
	# Optimize: back
	W.grad = None
	loss.backward()
	# Optimize: adjust underlying weights
	W.data += -10 * W.grad
print("Final loss: ", loss)

# Generate new names
for i in range(10):
	input_idx = 0
	gen_name = []

	while True:
		input_enc = F.one_hot(torch.tensor([input_idx]), num_classes=27).float()
		logits = input_enc @ W
		counts = logits.exp()
		p = counts / counts.sum(1, keepdims=True)

		input_idx = torch.multinomial(p, num_samples=1, replacement=True).item()
		gen_name.append(index_to_string[input_idx])
		if input_idx == 0:
			break
	print(''.join(gen_name))





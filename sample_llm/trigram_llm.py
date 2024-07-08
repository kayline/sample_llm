import torch
import torch.nn.functional as F
# Read and format data
names  = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))
string_to_index = {s:i+1 for i,s in enumerate(chars)}
string_to_index['.'] = 0
index_to_string = {i:s for s,i in string_to_index.items()}
# Build trigrams
inputs1, labels1, inputs2, labels2 = [], [], [], []
for name in names:
	chars = ['.'] + list(name) + ['.']
	for first,second in zip(chars, chars[1:]):
		first_index  = string_to_index[first]
		second_index = string_to_index[second]
		inputs1.append(first_index)
		labels1.append(second_index)
	for second,third in zip(chars[1:], chars[2:]):
		second_index = string_to_index[second]
		third_index = string_to_index[third]

		inputs2.append(second_index)
		labels2.append(third_index)
inputs1 = torch.tensor(inputs1)
labels1 = torch.tensor(labels1)
inputs2 = torch.tensor(inputs2)
labels2 = torch.tensor(labels2)

num_inputs1 = inputs1.nelement()
num_labels1 = labels1.nelement()
num_inputs2 = inputs2.nelement()
num_labels2 = labels2.nelement()

# print(num_inputs1)
# print(num_labels1)
# print(num_inputs2)
# print(num_labels2)

# First layer
# Build initial 'random' weights matrix
W1 = torch.randn((27,27), requires_grad=True)
loss1 = 0
# Optimize
for j in range(100):
	# Optimize: generate some output
	inputs1_enc = F.one_hot(inputs1, num_classes=27).float()
	inputs1_nums = inputs1_enc @ W1
	inputs1_counts = inputs1_nums.exp()
	inputs1_probs = inputs1_counts / inputs1_counts.sum(1, keepdims=True)
	loss1 = -inputs1_probs[torch.arange(num_inputs1), labels1].log().mean() + + 0.01*(W1**2).mean()
	# print("Loss 1", loss1.item())
	# Optimize: back
	W1.grad = None
	loss1.backward()
	# Optimize: adjust underlying weights
	W1.data += -10 * W1.grad
print("Final loss 1: ", loss1.item())

# Second layer
# Build initial 'random' weights matrix
W2 = torch.randn((27,27), requires_grad=True)
loss2 = 0
# Optimize
for k in range(100):
	# Optimize: generate some output
	inputs2_enc = F.one_hot(inputs2, num_classes=27).float()
	inputs2_nums = inputs2_enc @ W2
	inputs2_counts = inputs2_nums.exp()
	inputs2_probs = inputs2_counts / inputs2_counts.sum(1, keepdims=True)
	loss2 = -inputs2_probs[torch.arange(num_inputs2), labels2].log().mean() + + 0.01*(W2**2).mean()
	# print("Loss 2: ",loss2.item())
	# Optimize: back
	W2.grad = None
	loss2.backward()
	# Optimize: adjust underlying weights
	W2.data += -10 * W2.grad
print("Final loss 2: ", loss2.item())	

# Generate new names
for i in range(10):
	input1_idx = 0
	input2_idx = 0
	gen_name = []

	while True:
		input1_enc = F.one_hot(torch.tensor([input1_idx]), num_classes=27).float()
		logits1 = input1_enc @ W1
		counts1 = logits1.exp()
		p1 = counts1 / counts1.sum(1, keepdims=True)

		input2_idx = torch.multinomial(p1, num_samples=1, replacement=True).item()
		gen_name.append(index_to_string[input2_idx])
		if input2_idx==0:
			break

		input2_enc = F.one_hot(torch.tensor([input2_idx]), num_classes=27).float()
		logits2 = input2_enc @ W2
		counts2 = logits2.exp()
		p2 = counts2 / counts2.sum(1, keepdims=True)
		input1_idx = torch.multinomial(p2, num_samples=1, replacement=True).item()
		gen_name.append(index_to_string[input1_idx])
		if input1_idx==0:
			break
	print(''.join(gen_name))





import torch

names  = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))
string_to_index = {s:i+1 for i,s in enumerate(chars)}
string_to_index['.'] = 0
index_to_string = {i:s for s,i in string_to_index.items()}
# Build model
follow_counts = torch.zeros((27,27), dtype=torch.int32)
for name in names:
	chars = ['.'] + list(name) + ['.']
	for ch1, ch2 in zip(chars, chars[1:]):
		ch1_index = string_to_index[ch1]
		ch2_index = string_to_index[ch2]
		follow_counts[ch1_index, ch2_index] += 1
follow_odds = (follow_counts + .5).float()
follow_odds /= follow_odds.sum(1, keepdims=True)
# Use model to generate name
gen_name = []
char_index = 0
while True:
	p = follow_odds[char_index]
	char_index = torch.multinomial(p, num_samples=1, replacement=True).item()
	gen_name.append(index_to_string[char_index])
	if char_index == 0:
		break
print(''.join(gen_name))
# Calculate model loss
log_liklihood = 0
count = 0
for name in names:
	chars = ['.'] + list(name) + ['.']
	for ch1, ch2 in zip(chars, chars[1:]):
		ch1_index = string_to_index[ch1]
		ch2_index = string_to_index[ch2]
		model_prob = follow_odds[ch1_index, ch2_index]
		log_liklihood += torch.log(model_prob)
		count += 1
		follow_counts[ch1_index, ch2_index] += 1
loss = -log_liklihood / count
print(f'{loss}')

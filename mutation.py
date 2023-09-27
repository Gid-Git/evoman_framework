def mutate(self, individual):
    # Applies bit mutation to the individual based on the mutation rate
    for i in range(len(individual)):
        if random.uniform(0, 1) < self.mutation_rate:
            # Convert the weight to binary representation
            binary_representation = list(format(int((individual[i] - self.dom_l) * (2**15) / (self.dom_u - self.dom_l)), '016b'))
            
            for j in range(len(binary_representation)):
                if random.uniform(0, 1) < self.mutation_rate:
                    binary_representation[j] = '1' if binary_representation[j] == '0' else '0'
            
            individual[i] = int("".join(binary_representation), 2) * (self.dom_u - self.dom_l) / (2**15) + self.dom_l
    return individual
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_map>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <numeric>

using namespace std;

typedef pair<int, int> Swap;
typedef vector<Swap> SwapSequence;
typedef vector<int> Bitstring;

vector<int> permute(const vector<int>& bits, const vector<int>& pattern) {
    vector<int> result(pattern.size());
    for (size_t i = 0; i < pattern.size(); i++) {
        result[i] = bits[pattern[i] - 1];
    }
    return result;
}

string bits_to_str(const vector<int>& bits) {
    string result;
    for (int bit : bits) {
        result += to_string(bit);
    }
    return result;
}

vector<int> apply_swap_sequence(vector<int> bits, const SwapSequence& swap_sequence) {
    for (const auto& swap : swap_sequence) {
        int i = swap.first;
        int j = swap.second;
        if (i != j) {
            std::swap(bits[i], bits[j]);
        }
    }
    return bits;
}

double evaluate_permutation(const SwapSequence& swap_sequence, const vector<vector<int>>& test_inputs, 
                          const vector<int>& target_pattern, int n_qubits = 8) {
    int score = 0;
    int max_score = test_inputs.size() * n_qubits;
    
    for (const auto& test_input : test_inputs) {
        vector<int> result = apply_swap_sequence(test_input, swap_sequence);
        vector<int> expected = permute(test_input, target_pattern);
        
        for (int i = 0; i < n_qubits; i++) {
            if (result[i] == expected[i]) {
                score++;
            }
        }
    }
    
    return static_cast<double>(score) / max_score;
}

bool verify_all_inputs(const SwapSequence& swap_sequence, const vector<int>& target_pattern, int n_qubits = 8) {
    int failures = 0;
    vector<string> failed_inputs;
    
    for (int i = 0; i < (1 << n_qubits); i++) {
        vector<int> test_input;
        for (int j = 0; j < n_qubits; j++) {
            test_input.push_back((i >> j) & 1);
        }
        
        vector<int> result = apply_swap_sequence(test_input, swap_sequence);
        vector<int> expected = permute(test_input, target_pattern);
        
        if (result != expected) {
            failures++;
            if (failures <= 5) {
                failed_inputs.push_back(bits_to_str(test_input));
            }
        }
    }
    
    if (failures > 0) {
        cout << "Failed for " << failures << " inputs out of " << (1 << n_qubits) << endl;
        cout << "First failed inputs: ";
        for (const auto& input : failed_inputs) {
            cout << input << " ";
        }
        cout << endl;
        return false;
    }
    
    return true;
}

SwapSequence generate_random_individual(int n_qubits = 8, int max_swaps = 15) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> swap_count_dist(5, max_swaps);
    uniform_int_distribution<> qubit_dist(0, n_qubits - 1);
    
    int n_swaps = swap_count_dist(gen);
    SwapSequence swap_sequence;
    
    for (int i = 0; i < n_swaps; i++) {
        int a = qubit_dist(gen);
        int b;
        do {
            b = qubit_dist(gen);
        } while (b == a);
        
        swap_sequence.push_back({a, b});
    }
    
    return swap_sequence;
}

SwapSequence crossover(const SwapSequence& parent1, const SwapSequence& parent2) {
    random_device rd;
    mt19937 gen(rd());
    
    if (parent1.empty() || parent2.empty()) {
        return parent1.empty() ? parent2 : parent1;
    }
    
    uniform_int_distribution<> crossover_point_dist(1, min(parent1.size(), parent2.size()) - 1);
    uniform_real_distribution<> choice_dist(0.0, 1.0);
    
    int crossover_point = crossover_point_dist(gen);
    
    SwapSequence child1, child2;
    
    child1.insert(child1.end(), parent1.begin(), parent1.begin() + crossover_point);
    child1.insert(child1.end(), parent2.begin() + crossover_point, parent2.end());
    
    child2.insert(child2.end(), parent2.begin(), parent2.begin() + crossover_point);
    child2.insert(child2.end(), parent1.begin() + crossover_point, parent1.end());
    
    return choice_dist(gen) < 0.5 ? child1 : child2;
}

SwapSequence mutate(SwapSequence individual, int n_qubits = 8, double mutation_rate = 0.2) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> chance_dist(0.0, 1.0);
    uniform_int_distribution<> qubit_dist(0, n_qubits - 1);
    
    for (size_t i = 0; i < individual.size(); i++) {
        if (chance_dist(gen) < mutation_rate) {
            int a, b;
            do {
                a = qubit_dist(gen);
                b = qubit_dist(gen);
            } while (a == b);
            
            individual[i] = {a, b};
        }
    }
    
    if (chance_dist(gen) < 0.2 && individual.size() < 20) {
        int a, b;
        do {
            a = qubit_dist(gen);
            b = qubit_dist(gen);
        } while (a == b);
        
        uniform_int_distribution<> insert_pos_dist(0, individual.size());
        int insert_pos = insert_pos_dist(gen);
        
        individual.insert(individual.begin() + insert_pos, {a, b});
    }
    
    if (chance_dist(gen) < 0.2 && individual.size() > 3) {
        uniform_int_distribution<> remove_pos_dist(0, individual.size() - 1);
        int remove_pos = remove_pos_dist(gen);
        
        individual.erase(individual.begin() + remove_pos);
    }
    
    return individual;
}

void trace_permutation_effect(vector<int>& state, const vector<int>& pattern) {
    vector<int> result(8, 0);
    for (int i = 0; i < 8; i++) {
        result[i] = state[pattern[i] - 1];
    }
    state = result;
}

SwapSequence optimize_gate_count(SwapSequence sequence) {
    for (size_t i = 0; i < sequence.size(); i++) {
        for (size_t j = i + 1; j < sequence.size(); j++) {
            if (sequence[i] == sequence[j]) {
                sequence.erase(sequence.begin() + j);
                j--;
            }
        }
    }
    
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 0; i < sequence.size() - 1; i++) {
            if (sequence[i] == sequence[i+1]) {
                sequence.erase(sequence.begin() + i, sequence.begin() + i + 2);
                improved = true;
                break;
            }
        }
    }
    
    return sequence;
}

SwapSequence inverse_sequence(const SwapSequence& sequence) {
    SwapSequence inverse;
    for (auto it = sequence.rbegin(); it != sequence.rend(); ++it) {
        inverse.push_back(*it);
    }
    return inverse;
}

SwapSequence brute_force_search(const vector<int>& target_pattern, int n_qubits = 8, int max_trials = 5000) {
    random_device rd;
    mt19937 gen(rd());
    
    SwapSequence best_sequence;
    double best_score = 0;
    
    vector<vector<int>> test_inputs;
    test_inputs.reserve(256);
    for (int i = 0; i < 256; i++) {
        vector<int> input;
        for (int j = 0; j < n_qubits; j++) {
            input.push_back((i >> j) & 1);
        }
        test_inputs.push_back(input);
    }
    
    cout << "Starting comprehensive brute force search..." << endl;
    
    for (int trial = 0; trial < max_trials; trial++) {
        SwapSequence candidate = generate_random_individual(n_qubits, n_qubits * 2);
        double score = evaluate_permutation(candidate, test_inputs, target_pattern, n_qubits);
        
        if (score > best_score) {
            best_score = score;
            best_sequence = candidate;
            
            cout << "Trial " << trial << ": New best score = " << best_score << endl;
            
            if (best_score == 1.0) {
                cout << "Found perfect solution with score 1.0!" << endl;
                return best_sequence;
            }
        }
        
        if (trial % 500 == 0) {
            cout << "Trial " << trial << ": Best score = " << best_score << endl;
        }
    }
    
    return best_sequence;
}

pair<SwapSequence, double> genetic_algorithm(const vector<int>& target_pattern, int n_qubits = 8, 
                                           int population_size = 100, int generations = 100,
                                           int elite_size = 10, double mutation_rate = 0.2, 
                                           int tournament_size = 3) {
    random_device rd;
    mt19937 gen(rd());
    
    vector<vector<int>> test_inputs;
    for (int i = 0; i < 256; i++) {
        vector<int> input;
        for (int j = 0; j < n_qubits; j++) {
            input.push_back((i >> j) & 1);
        }
        test_inputs.push_back(input);
    }
    
    cout << "Using all 256 test inputs for genetic algorithm" << endl;
    
    vector<SwapSequence> population;
    for (int i = 0; i < population_size; i++) {
        population.push_back(generate_random_individual(n_qubits));
    }
    
    SwapSequence best_individual;
    double best_score = 0;
    
    vector<double> best_scores;
    vector<double> avg_scores;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    for (int generation = 0; generation < generations; generation++) {
        vector<double> scores;
        for (const auto& ind : population) {
            scores.push_back(evaluate_permutation(ind, test_inputs, target_pattern, n_qubits));
        }
        
        double gen_best_score = *max_element(scores.begin(), scores.end());
        double gen_avg_score = accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        
        best_scores.push_back(gen_best_score);
        avg_scores.push_back(gen_avg_score);
        
        if (gen_best_score > best_score) {
            best_score = gen_best_score;
            auto best_idx = max_element(scores.begin(), scores.end()) - scores.begin();
            best_individual = population[best_idx];
            
            cout << "Generation " << generation << ": New best score = " << best_score << endl;
            
            if (best_score == 1.0) {
                cout << "Found perfect solution with score 1.0!" << endl;
                break;
            }
        }
        
        if (generation % 10 == 0 || generation == generations - 1) {
            auto elapsed = chrono::duration_cast<chrono::seconds>(
                chrono::high_resolution_clock::now() - start_time).count();
            
            cout << "Generation " << generation << ": Best score = " << gen_best_score 
                 << ", Avg score = " << gen_avg_score << ", Time: " << elapsed << "s" << endl;
        }
        
        vector<int> elite_indices(scores.size());
        iota(elite_indices.begin(), elite_indices.end(), 0);
        sort(elite_indices.begin(), elite_indices.end(), 
             [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });
        
        vector<SwapSequence> elites;
        for (int i = 0; i < elite_size; i++) {
            elites.push_back(population[elite_indices[i]]);
        }
        
        vector<SwapSequence> new_population = elites;
        
        uniform_int_distribution<> pop_dist(0, population_size - 1);
        
        while (new_population.size() < population_size) {
            vector<int> tournament1, tournament2;
            for (int i = 0; i < tournament_size; i++) {
                tournament1.push_back(pop_dist(gen));
                tournament2.push_back(pop_dist(gen));
            }
            
            int parent1_idx = *max_element(tournament1.begin(), tournament1.end(), 
                                         [&scores](int i1, int i2) { return scores[i1] < scores[i2]; });
            
            int parent2_idx = *max_element(tournament2.begin(), tournament2.end(), 
                                         [&scores](int i1, int i2) { return scores[i1] < scores[i2]; });
            
            SwapSequence parent1 = population[parent1_idx];
            SwapSequence parent2 = population[parent2_idx];
            
            SwapSequence child = crossover(parent1, parent2);
            child = mutate(child, n_qubits, mutation_rate);
            
            new_population.push_back(child);
        }
        
        population = new_population;
    }
    
    if (best_score == 1.0) {
        best_individual = optimize_gate_count(best_individual);
    }
    
    return {best_individual, best_score};
}

void optimize_permutation(const string& name, const vector<int>& pattern) {
    cout << "========== Optimizing " << name << " ==========" << endl;
    
    auto [ga_result, ga_score] = genetic_algorithm(pattern, 8, 200, 500, 20, 0.2, 5);
    
    bool ga_verified = (ga_score == 1.0);
    
    if (!ga_verified) {
        cout << "\nGenetic algorithm did not find perfect solution. Trying brute force search..." << endl;
        SwapSequence bf_result = brute_force_search(pattern);
        
        double bf_score = evaluate_permutation(bf_result, {{1,0,1,0,1,0,1,0}}, pattern);
        cout << "Brute force search result score: " << bf_score << endl;
        
        if (bf_score == 1.0) {
            ga_result = bf_result;
            ga_verified = true;
        }
    }
    
    cout << "\nFinal sequence for " << name << ":" << endl;
    for (auto [i, j] : ga_result) {
        cout << "circuit.swap(qubits[" << i << "], qubits[" << j << "]);" << endl;
    }
    
    cout << "\nInverse " << name << ":" << endl;
    SwapSequence inverse = inverse_sequence(ga_result);
    for (auto [a, b] : inverse) {
        cout << "circuit.swap(qubits[" << a << "], qubits[" << b << "]);" << endl;
    }
    
    cout << "\n" << endl;
}

int main() {
    vector<int> ip = {2, 6, 3, 1, 4, 8, 5, 7};
    vector<int> fp = {4, 1, 3, 5, 7, 2, 8, 6};
    
    optimize_permutation("Initial Permutation", ip);
    optimize_permutation("Final Permutation", fp);
    
    return 0;
}
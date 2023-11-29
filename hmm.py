from hmmlearn import hmm

# Dữ liệu huấn luyện mẫu
train_data = [
    ['I', 'love', 'coding'],
    ['Python', 'is', 'great'],
    ['Machine', 'learning', 'is', 'interesting']
]

# Khởi tạo mô hình HMM
model = hmm.MultinomialHMM(n_components=2)

# Xây dựng ma trận đặc trưng
features = []
for sentence in train_data:
    feature = [word.lower() for word in sentence]
    features.append(feature)

# Huấn luyện mô hình
model.fit(features)

# Dự đoán trạng thái của một câu mới
new_sentence = ['I', 'enjoy', 'programming']
test_feature = [word.lower() for word in new_sentence]
observed_sequence = [model.n_features_ - 1] * len(test_feature)  # Dùng trạng thái cuối cùng làm giả
observed_sequence.extend(test_feature)
observed_sequence = np.array([observed_sequence]).reshape(1, -1)

predicted_states = model.predict(observed_sequence)
print("Predicted States:", predicted_states)

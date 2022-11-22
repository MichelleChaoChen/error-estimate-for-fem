from neural_network import Neural_NetWork

if __name__ == '__main__':
    train_features = [
        [1,2,3],
        [2,3,4]
    ]

    train_labels = [0,1]

    model = Neural_NetWork()
    
    model.train(train_features, train_labels)

def plo_class_histg1(image_feats, image_labels, kmeans):
    vocab_size = kmeans.cluster_centers_.shape[0]
    histogram = np.zeros((15, vocab_size))
    counting = np.zeros((15, 1))
    for i in len(image_feats):
        label = image_labels[i]
        histogram[label] += image_feats[i]
        counting += 1
    for j, his in enumerate(histogram):
        his = his/counting[j]
        index = np.arange(50)
        plt.figure()
        plt.bar(index, his)
    plt.show()
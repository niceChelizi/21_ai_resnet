metric = model.eval(de_test,dataset_sink_mode=False)
print('整体：')
print(metric)

classdic={}
for i in dic.keys():
    classdic[dic[i]]=i
print(classdic)
for i in range(10):
    test_ = de_train.create_dict_iterator().__next__()
    test = Tensor(test_['image'], mindspore.float32)
    predictions = model.predict(test)
    predictions = predictions.asnumpy()
    true_label = test_['label'].asnumpy()
    p_np = predictions[0, :]
    pre_label = np.argmax(p_np)
    print('第' + str(i) + '个sample预测结果：', classdic[pre_label], '   真实结果：', classdic[true_label[0]])
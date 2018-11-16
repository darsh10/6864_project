import api
import json

if __name__ == "__main__":
    engine = api.QRAPI("model2.pkl.gz","../../../tao_model.txt","../../../apple-text-vector.txt")
    corpus = open("../../../tao_model.txt")
    d = {}
    d2 = {}
    for line in corpus:
        id, title, body = line.strip().split("\t")
        d[id] = (title, body)
        d2[(title, body)] = id
    test = open("../../../test_50_docs.txt")
    actual = []
    pred = []
    for line in test:
        query_id, pos_cand_id, cand_list_ids = line.strip().split("\t")
        cand_list_ids = cand_list_ids.split()
        cand_list_data = []
        for cand_id in cand_list_ids:
            cand_list_data.append(d[cand_id])
        query_test_data = {'query':d[query_id],'candidates':cand_list_data}
        results = engine.rank(json.dumps(query_test_data))
        actual.append(pos_cand_id)
        pred.append(d2[cand_list_data[results["ranks"][0]]])
        print actual[-1], pred[-1]
    import IPython; IPython.embed()

import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F
import pickle
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def test(params, model, testset, category, txt_processors):
    model.eval()
    if category == 'fashion200k':
        model.eval()
        test_queries = testset.get_test_queries()
        with torch.no_grad():
            all_imgs = []
            all_captions = []
            all_queries = []
            all_target_captions = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [testset.get_img(t['source_img_id'])]
                    mods += [t['mod']['str']]
                    if len(imgs) >= params.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float()
                        imgs = torch.autograd.Variable(imgs).cuda()
                        mods = [t.encode('utf-8').decode('utf-8') for t in mods]
                        f = model.extract_retrieval_compose(imgs, mods)
                        f = f.data.cpu().numpy()
                        all_queries += [f]
                        imgs = []
                        mods = []
                all_queries = np.concatenate(all_queries)
                all_target_captions = [t['target_caption'] for t in test_queries]
                # compute all image features
                imgs = []
                for i in tqdm(range(len(testset.imgs))):
                #for i in tqdm(range(100)):
                    imgs += [testset.get_img(i)]
                    if len(imgs) >= params.batch_size or i == len(testset.imgs) - 1:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float()
                        imgs = torch.autograd.Variable(imgs).cuda()
                        imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
                        all_imgs += [imgs]
                        imgs = []
                all_imgs = np.concatenate(all_imgs)
                all_captions = [img['captions'][0] for img in testset.imgs]
            # feature normalization
            for i in range(all_queries.shape[0]):
                all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
            for i in range(all_imgs.shape[0]):
                all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
            
            # match test queries to target images, get nearest neighbors
            sims = all_queries.dot(all_imgs.T)
            if test_queries:
                for i, t in enumerate(test_queries):
                    sims[i, t['source_img_id']] = -10e10  # remove query image
            nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

            # compute recalls
            out = []
            nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
            for k in [1, 10, 50]:
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if all_target_captions[i] in nns[:k]:
                        r += 1
                r /= len(nn_result)
                out += [('recall_top' + str(k) + '_correct_composition', r * 100)]

            return out

    else:
        if category == 'dress':
            (test_queries, test_targets, name) = (testset.test_queries_dress, testset.test_targets_dress, 'dress')
        elif category == 'shirt':
            (test_queries, test_targets, name) = (testset.test_queries_shirt, testset.test_targets_shirt, 'shirt')
        elif category == 'toptee':
            (test_queries, test_targets, name) = (testset.test_queries_toptee, testset.test_targets_toptee, 'toptee')
        elif category == 'shoes':
            (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, 'shoes')
        elif category == 'lasco':
            (test_queries, test_targets) = testset.get_val_queries()
            name = "lasco"
        elif category == 'birds':
            test_queries = testset.get_test_queries()
            test_targets = testset.get_test_targets()
            name = 'birds'

        with torch.no_grad():
            all_queries = []
            all_imgs = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [t['source_img_data']]
                    mods += [t['mod']['str']]
                    if len(imgs) >= params.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        mods = [txt_processors["eval"](caption) for caption in mods]
                        f = model.extract_retrieval_compose(imgs, mods)
                        f = f.data.cpu().numpy()
                        all_queries += [f]
                        imgs = []
                        mods = []
                all_queries = np.concatenate(all_queries)

                # compute all image features
                imgs = []
                logits = []
                for t in tqdm(test_targets):
                    imgs += [t['target_img_data']]
                    if len(imgs) >= params.batch_size or t is test_targets[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
                        all_imgs += [imgs]
                        imgs = []
                all_imgs = np.concatenate(all_imgs)

        # feature normalization
        # for i in range(all_queries.shape[0]):
        #     all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
        # for i in range(all_imgs.shape[0]):
        #     all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
        
        
        # match test queries to target images, get nearest neighbors
        sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
        sims = sims.squeeze()
        sims = sims.max(-1)

        
        test_targets_id = []
        for i in test_targets:
            test_targets_id.append(i['target_img_id'])
        for i, t in enumerate(test_queries):
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

        # compute recalls
        out = []
        if category=="lasco":
            nn_result = [np.argsort(-sims[i, :])[:1000] for i in range(sims.shape[0])]
            for k in [1, 5, 10, 50, 500, 1000]:
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                        r += 1
                r = 100 * r / len(nn_result)
                out += [('{}_r{}'.format(name, k), r)]
        else:
            nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
            for k in [1, 10, 50]:
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                        r += 1
                r = 100 * r / len(nn_result)
                out += [('{}_r{}'.format(name, k), r)]

        return out

def lasco_query(params, model):
    test_queries = load_obj("/root/autodl-tmp/LaSCo/lasco_val_queries.pkl")
    all_queries = []
    imgs = []
    mods = []
    for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
        imgs += [t['source_img_da ta']]
        mods += [t['mod']['str']]
        if len(imgs) >= params.batch_size or t is test_queries[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            f = model.extract_retrieval_compose(imgs, mods)
            f = f.data.cpu().numpy()
            all_queries += [f]
            imgs = []
            mods = []
    all_queries = np.concatenate(all_queries)
    return all_queries

def lasco_target(params, model):
    test_targets = load_obj("/root/autodl-tmp/LaSCo/lasco_val_targets.pkl")
    all_imgs = []
    imgs = []
    logits = []
    for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
        imgs += [t['target_img_data']]
        if len(imgs) >= params.batch_size or t is test_targets[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
            all_imgs += [imgs]
            imgs = []
    all_imgs = np.concatenate(all_imgs)
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    return all_imgs, test_targets_id

def test_lasco(params, model, testset=None, category=None):
    model.eval()

    with torch.no_grad():
        all_imgs, test_targets_id = lasco_target(params, model)
        test_queries = load_obj("/root/autodl-tmp/LaSCo/lasco_val_queries.pkl")
        all_queries = []
        imgs = []
        mods = []
        for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
            imgs += [t['source_img_data']]
            mods += [t['mod']['str']]
            if len(imgs) >= params.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float().cuda()
                f = model.extract_retrieval_compose(imgs, mods)
                f = f.data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)



    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    # test_targets_id = []
    # for i in test_targets:
    #     test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10



    # compute recalls
    out = []
    nn_result = [np.argsort(-sims[i, :])[:1000] for i in range(sims.shape[0])]
    for k in [1, 5, 10, 50, 500, 1000]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format("lasco", k), r)]

    return out



def test_cirr_valset(params, model, testset, txt_processors):
    test_queries, test_targets = testset.val_queries, testset.val_targets
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in tqdm(test_queries):
                imgs += [t['source_img_data']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    mods = [txt_processors["eval"](caption) for caption in mods]
                    f = model.extract_retrieval_compose(imgs, mods).data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    # for i in range(all_queries.shape[0]):
    #     all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    # for i in range(all_imgs.shape[0]):
    #     all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    # sims = all_queries.dot(all_imgs.T)
    sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
    sims = sims.squeeze()
    sims = sims.max(-1)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] # (m,n)

    # all set recalls
    cirr_out = []
    for k in [1, 5, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_r{}'.format(params.dataset,k), r)]

    # subset recalls
    for k in [1, 2, 3]:
        r = 0.0
        for i, nns in enumerate(nn_result):

            subset = np.array([test_targets_id.index(idx) for idx in test_queries[i]['subset_id']]) # (6)
            subset_mask = (nns[..., None] == subset[None, ...]).sum(-1).astype(bool) # (n,1)==(1,6) => (n,6) => (n)
            subset_label = nns[subset_mask] # (6)
            if test_targets_id.index(test_queries[i]['target_img_id']) in subset_label[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_subset_r{}'.format(params.dataset, k), r)]

    return cirr_out

import pickle, os, sys
p=r'C:\Users\Acer\Desktop\mini_project 22\mini_project 22\data\landmarks\landmarks_custom_both_hands_face.pkl'
print('path=',p)
print('exists', os.path.exists(p))
with open(p,'rb') as f:
    data=pickle.load(f)
print('type', type(data))
if isinstance(data, dict):
    keys=list(data.keys())
    print('num keys', len(keys))
    for k in keys[:50]:
        try:
            print(repr(k), '->', type(data[k]), 'len=', len(data[k]))
        except Exception as e:
            print('key', repr(k), '->', type(data[k]), 'err', e)
else:
    try:
        print('len', len(data))
        print('sample type', type(data[0]))
    except Exception as e:
        print('cannot introspect:', e)

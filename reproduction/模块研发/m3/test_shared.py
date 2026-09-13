import unittest
from shared_screen import select,GROUPS


class SharedTests(unittest.TestCase):
    def test_retain_parent_and_shared_selection(self):
        trials=[]
        for g in GROUPS:
            for w in [0,.5,1]:
                trials.append(dict(group=g,key='model',weight=w,validation_mae=1-w))
        for policy in ['balanced','full']:
            self.assertEqual(select(trials,policy)['weight'],.5)

    def test_identity_tie(self):
        trials=[dict(group=g,key=k,weight=w,validation_mae=1.) for g in GROUPS for k in ['a','b'] for w in [0,.5]]
        for policy in ['balanced','full']:self.assertEqual(select(trials,policy)['weight'],0)


if __name__=='__main__':unittest.main()

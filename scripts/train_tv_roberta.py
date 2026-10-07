"""Bounded CUDA-only, one-epoch TV adaptation; no test data or hub uploads."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

MODEL = 'Zritze/imdb-spoiler-robertaOrigDatasetLR1'
REVISION = '56fee120f8495ccfc5001e3fbd1478656d17001a'


def metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score, matthews_corrcoef, average_precision_score
    y = np.asarray(labels); p = np.asarray(scores) >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(rows=len(y), positives=int(y.sum()), threshold=.5, precision=float(precision), recall=float(recall), f1=float(f1), accuracy=float((y == p).mean()), balanced_accuracy=float(balanced_accuracy_score(y,p)), mcc=float(matthews_corrcoef(y,p)), average_precision=float(average_precision_score(y,scores)), false_positive_rate=float(fp/(fp+tn)) if fp+tn else None, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def load_pack(root):
    manifest=json.loads((root/'experiment.json').read_text())
    rows={}
    for split in ['train','validation']:
        path=root/(split+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['hashes'][path.name], 'Dataset changed'
        rows[split]=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert rows[split] and len({r['id'] for r in rows[split]})==len(rows[split])
        assert all(type(r['label']) is int and r['label'] in [0,1] and 0<len(r['input_ids'])<=512 for r in rows[split])
    assert {r['work_page'] for r in rows['train']}.isdisjoint({r['work_page'] for r in rows['validation']})
    assert {r['id'] for r in rows['train']}.isdisjoint({r['id'] for r in rows['validation']})
    assert {r['text'].strip().casefold() for r in rows['train']}.isdisjoint({r['text'].strip().casefold() for r in rows['validation']})
    return rows,manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=['sentence','context'],required=True)
    parser.add_argument('--allow-exploratory-corpus-labels',action='store_true')
    args=parser.parse_args()
    if not args.allow_exploratory_corpus_labels:
        raise SystemExit('Pack has unverified markup labels/context; explicit exploratory flag required.')
    os.environ['WANDB_DISABLED']='true';os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    import torch
    assert torch.cuda.is_available(), 'CUDA GPU required. No CPU/Mac training fallback.'
    import numpy as np
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    rows,manifest=load_pack(args.root)
    out=args.root/'results'/args.arm;out.mkdir(parents=True,exist_ok=True)
    run_config=dict(model=MODEL,revision=REVISION,arm=args.arm,seed=42,epochs=1,learning_rate=2e-5,batch_size=8,gradient_accumulation=2,max_length=512,threshold=.5,pack_hashes=manifest['hashes'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__,transformers=transformers.__version__,exploratory=True)
    configpath=out/'run-config.json'
    if configpath.exists():assert json.loads(configpath.read_text())==run_config,'Run configuration changed; use a new output directory'
    configpath.write_text(json.dumps(run_config,indent=2)+'\n')
    if (out/'report.json').exists():
        print('Already completed; not training again:',args.arm,flush=True);return
    set_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,use_safetensors=True)
    assert model.config.num_labels==2
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,data):
            self.items=[]
            for row in data:
                # Identical sentence component in both arms; context arm uses frozen pair IDs.
                enc=tokenizer(row['sentence_input'],truncation=True,max_length=256) if args.arm=='sentence' else dict(input_ids=row['input_ids'],attention_mask=row['attention_mask'])
                self.items.append(dict(**enc,labels=row['label']))
        def __len__(self):return len(self.items)
        def __getitem__(self,i):return self.items[i]
    train=Dataset(rows['train']);validation=Dataset(rows['validation'])
    counts=Counter(r['label'] for r in rows['train'])
    weights=torch.tensor([len(train)/(2*counts[k]) for k in [0,1]],dtype=torch.float32)
    class WeightedTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels=inputs['labels'];outputs=model(**{k:v for k,v in inputs.items() if k!='labels'})
            loss=torch.nn.functional.cross_entropy(outputs.logits.float(),labels,weight=weights.to(outputs.logits.device))
            return (loss,outputs) if return_outputs else loss
    def compute_metrics(prediction):
        logits=prediction.predictions
        if isinstance(logits,tuple):logits=logits[0]
        scores=torch.softmax(torch.as_tensor(logits),dim=-1)[:,1].numpy()
        return metrics(prediction.label_ids,scores)
    training_args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=1,learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=16,gradient_accumulation_steps=2,weight_decay=.01,warmup_ratio=.1,fp16=True,eval_strategy='epoch',save_strategy='steps',save_steps=25,save_total_limit=1,logging_steps=5,report_to=[],seed=42,data_seed=42,dataloader_num_workers=0,save_safetensors=True,disable_tqdm=True)
    trainer=WeightedTrainer(model=model,args=training_args,train_dataset=train,eval_dataset=validation,data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),compute_metrics=compute_metrics)
    checkpoints=sorted((out/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    beforepath=out/'before.json'
    if not beforepath.exists():beforepath.write_text(json.dumps(trainer.evaluate(),indent=2)+'\n')
    started=time.monotonic()
    result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    prediction=trainer.predict(validation)
    scores=torch.softmax(torch.as_tensor(prediction.predictions),dim=-1)[:,1].numpy()
    records=[dict(id=r['id'],work_page=r['work_page'],label=r['label'],score=float(s),prediction=int(s>=.5),episode_selection_method=r['episode_selection_method']) for r,s in zip(rows['validation'],scores)]
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    trainer.save_model(str(out/'model'));tokenizer.save_pretrained(out/'model')
    report=dict(config=run_config,train_class_weights=weights.tolist(),before=json.loads(beforepath.read_text()),after=metrics([r['label'] for r in records],scores),training_metrics=result.metrics,weighted_validation_loss=prediction.metrics.get('test_loss'),elapsed_training_and_save_seconds=time.monotonic()-started,per_work={w:metrics([r['label'] for r in records if r['work_page']==w],[r['score'] for r in records if r['work_page']==w]) for w in sorted({r['work_page'] for r in records})},always_spoiler=metrics([r['label'] for r in records],[1.]*len(records)),warning='Exploratory markup-label agreement on 3 validation shows. Not product gold or held-out test. Context grounding incomplete. One seed/epoch; no threshold fitting.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TV EPOCH COMPLETE',args.arm,json.dumps(report),flush=True)

if __name__=='__main__':main()

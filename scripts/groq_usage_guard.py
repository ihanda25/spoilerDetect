"""Experiment-local cumulative request/token guard for the existing urllib runner."""
import io
import json
from pathlib import Path
from urllib.error import HTTPError

class BudgetStop(RuntimeError):pass

class Response(io.BytesIO):
    def __init__(self,body,headers):super().__init__(body);self.headers=headers

class Guard:
    def __init__(self,runner,out,max_requests=120,max_tokens=100000):
        self.runner=runner;self.out=out;self.original=runner.urlopen;self.path=out/'usage-budget.json'
        if self.path.exists():self.state=json.loads(self.path.read_text())
        else:
            cache=json.loads((out/'rag.json').read_text());status=json.loads((out/'status.json').read_text())
            predictions=cache['predictions'];tokens=sum((p.get('usage') or {}).get('total_tokens',0) for p in predictions.values())
            attempts=status.get('api_requests_this_invocation',len(predictions))
            # Previous failed request lacked usage; reserve a worst-case byte-token bound.
            unknown=max(0,attempts-len(predictions));reserve=max((len(json.dumps(dict(title=r['title'],excerpt=r['excerpt'],public_premise=r['public_premise'],passages=r['passages'])).encode())+len(runner.SYSTEM.encode())+1200 for r in json.loads((out/'inputs.json').read_text())),default=0)
            self.state=dict(request_attempts=attempts,actual_reported_tokens=tokens,budget_charged_tokens=tokens+unknown*reserve,unknown_usage_attempts=unknown,events=[],prior_attempts_imported=True)
        self.state.update(max_request_attempts=max_requests,max_total_tokens=max_tokens,policy='Cumulative across resumes. Preflight UTF-8 byte upper bound plus max completion; errors without usage charged conservatively. No paid fallback.')
        self.save()
    def save(self):self.runner.save(self.path,self.state)
    def open(self,request,timeout=60):
        payload=json.loads(request.data)
        reserve=len(json.dumps(payload['messages']).encode())+payload['max_completion_tokens']+128
        if self.state['request_attempts']>=self.state['max_request_attempts']:raise BudgetStop('cumulative request-attempt cap reached')
        if self.state['budget_charged_tokens']+reserve>self.state['max_total_tokens']:raise BudgetStop('cumulative token cap: insufficient reserve for next request')
        self.state['request_attempts']+=1;self.save()
        try:
            with self.original(request,timeout=timeout) as response:body=response.read();headers=response.headers
            data=json.loads(body);usage=data.get('usage') or {};actual=usage.get('total_tokens')
            if actual is None:self.state['unknown_usage_attempts']+=1
            else:self.state['actual_reported_tokens']+=actual
            self.state['budget_charged_tokens']+=reserve if actual is None else actual
            self.state['last_rate_limit_headers']={k:headers.get(k) for k in ['x-ratelimit-limit-tokens','x-ratelimit-remaining-tokens','x-ratelimit-remaining-requests','x-ratelimit-reset-tokens','x-ratelimit-reset-requests','retry-after'] if headers.get(k) is not None}
            self.save();return Response(body,headers)
        except HTTPError as exc:
            body=exc.read();self.state['unknown_usage_attempts']+=1;self.state['budget_charged_tokens']+=reserve
            try:error=json.loads(body).get('error',{})
            except (ValueError,AttributeError):error={}
            token=self.runner.key()
            message=str(error.get('message','')).replace(token,'[REDACTED]') if token else str(error.get('message',''))
            self.state['events'].append(dict(http_status=exc.code,code=error.get('code'),type=error.get('type'),message=message[:1500],rate_limit_headers={k:exc.headers.get(k) for k in ['retry-after','x-ratelimit-remaining-requests','x-ratelimit-remaining-tokens','x-ratelimit-reset-tokens','x-ratelimit-reset-requests'] if exc.headers.get(k) is not None}))
            self.save();raise HTTPError(exc.url,exc.code,exc.msg,exc.headers,io.BytesIO(body)) from None

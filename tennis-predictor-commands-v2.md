# Tennis Predictor — AWS Command Reference

## START THE API

**1. Launch the task:**
```bash
aws ecs run-task \
  --cluster tennis-predictor \
  --task-definition tennis-predictor:3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-057ebc5529e804bdd],securityGroups=[sg-0e386b9e56fc3489f],assignPublicIp=ENABLED}" \
  --region us-east-1
```

**2. Wait 60 seconds, then get the public IP:**
```bash
aws ec2 describe-network-interfaces \
  --network-interface-ids $(aws ecs describe-tasks \
    --cluster tennis-predictor \
    --tasks $(aws ecs list-tasks --cluster tennis-predictor --region us-east-1 --query "taskArns[0]" --output text) \
    --region us-east-1 \
    --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
    --output text) \
  --query "NetworkInterfaces[0].Association.PublicIp" \
  --output text \
  --region us-east-1
```

**3. Test it:**
```bash
curl http://<PUBLIC_IP>/health
```

---

## STOP THE API

```bash
aws ecs stop-task \
  --cluster tennis-predictor \
  --task $(aws ecs list-tasks --cluster tennis-predictor --region us-east-1 --query "taskArns[0]" --output text) \
  --region us-east-1
```

---

## CHECK STATUS

**Is the task running?**
```bash
aws ecs list-tasks --cluster tennis-predictor --region us-east-1
```
Running = task ARN listed. Stopped = `"taskArns": []`

**Running tasks count:**
```bash
aws ecs describe-clusters \
  --clusters tennis-predictor \
  --region us-east-1 \
  --query "clusters[0].{running: runningTasksCount, pending: pendingTasksCount}" \
  --output json
```

---

## DEBUGGING

**Why did the task stop?**
```bash
aws ecs describe-tasks \
  --cluster tennis-predictor \
  --tasks $(aws ecs list-tasks --cluster tennis-predictor --desired-status STOPPED --region us-east-1 --query "taskArns[0]" --output text) \
  --region us-east-1 \
  --query "tasks[0].{status: lastStatus, stopCode: stopCode, reason: stoppedReason, container: containers[0].reason}" \
  --output json
```

**View live container logs (replace TASK_ID):**
```bash
aws logs get-log-events \
  --log-group-name /ecs/tennis-predictor \
  --log-stream-name ecs/tennis-predictor/<TASK_ID> \
  --region us-east-1 \
  --query "events[-20:].message" \
  --output text
```

**Get TASK_ID of last stopped task:**
```bash
aws ecs list-tasks \
  --cluster tennis-predictor \
  --desired-status STOPPED \
  --region us-east-1
```

---

## UPDATE AFTER CODE CHANGES

**1. Rebuild and push image:**
```bash
docker buildx build --platform linux/amd64 -t pakich17/tennis-predictor:latest --push .
```

**2. Stop current task:**
```bash
aws ecs stop-task \
  --cluster tennis-predictor \
  --task $(aws ecs list-tasks --cluster tennis-predictor --region us-east-1 --query "taskArns[0]" --output text) \
  --region us-east-1
```

**3. Relaunch (ECS always pulls latest image on start):**
```bash
aws ecs run-task \
  --cluster tennis-predictor \
  --task-definition tennis-predictor:3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-057ebc5529e804bdd],securityGroups=[sg-0e386b9e56fc3489f],assignPublicIp=ENABLED}" \
  --region us-east-1
```

---

## SECURITY GROUP — UPDATE YOUR IP

**Check your current IP:**
```bash
curl -s https://checkip.amazonaws.com
```

**Add a new IP:**
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-0e386b9e56fc3489f \
  --protocol tcp \
  --port 80 \
  --cidr <YOUR_IP>/32 \
  --region us-east-1
```

**Remove an old IP:**
```bash
aws ec2 revoke-security-group-ingress \
  --group-id sg-0e386b9e56fc3489f \
  --protocol tcp \
  --port 80 \
  --cidr <OLD_IP>/32 \
  --region us-east-1
```

**View all current rules:**
```bash
aws ec2 describe-security-groups \
  --group-ids sg-0e386b9e56fc3489f \
  --region us-east-1 \
  --query "SecurityGroups[0].IpPermissions" \
  --output json
```

---

## UPDATE FRONTEND (after API IP changes)

**1. Remove the old env variable:**
```bash
vercel env rm ECS_API_URL production
```

**2. Add the new IP:**
```bash
vercel env add ECS_API_URL production
# paste: http://YOUR_NEW_IP
```

**3. Redeploy to production:**
```bash
vercel --prod
```

---

## RESEED DATABASE (after new CSV data)

```bash
cd ~/Documents/Tennis-Predictor
source venv/bin/activate
export DATABASE_URL="postgresql://postgres.vvnirrqkkyybivatujux:Chiquitul457622@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
python scripts/load_data.py
```

---

## TEST ENDPOINTS

**Health check:**
```bash
curl http://<PUBLIC_IP>/health
```

**Predict match:**
```bash
curl -X POST http://<PUBLIC_IP>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player1_name": "Djokovic",
    "player2_name": "Alcaraz",
    "surface": "Clay",
    "tournament_level": "G",
    "round": "F",
    "best_of": 5,
    "indoor": false
  }'
```

**Search players:**
```bash
curl "http://<PUBLIC_IP>/players?search=Nadal"
```

**H2H stats:**
```bash
curl "http://<PUBLIC_IP>/h2h?player1=Djokovic&player2=Alcaraz"
```

**Player ELO:**
```bash
curl "http://<PUBLIC_IP>/players/Djokovic/elo"
```

**Debug features:**
```bash
curl "http://<PUBLIC_IP>/debug/Djokovic?surface=Clay"
```

---

## KEY RESOURCE IDs (save these)

| Resource | ID |
|---|---|
| ECS Cluster | tennis-predictor |
| Task Definition | tennis-predictor:3 |
| Security Group | sg-0e386b9e56fc3489f |
| Subnet | subnet-057ebc5529e804bdd |
| VPC | vpc-05fcf99892b5a9121 |
| Docker Hub Image | pakich17/tennis-predictor:latest |
| Supabase DB | aws-1-us-east-1.pooler.supabase.com |



## Local Testing: # Terminal 1 — start API
cd ~/Documents/Tennis-Predictor
source venv/bin/activate
export DATABASE_URL="postgresql://postgres.vvnirrqkkyybivatujux:Chiquitul457622@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
uvicorn api.main:app --reload --port 8001

# Terminal 2 — test predictions
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"player1_name":"PLAYER1","player2_name":"PLAYER2","surface":"Clay","tournament_level":"G","round":"F","best_of":5,"indoor":false}' | python3 -m json.tool

# Or update frontend .env.local to point to localhost
# ECS_API_URL=http://localhost:8001
# then npm run dev in tennis-predictor-frontend
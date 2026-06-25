"""Run the video-builder deploy via boto3 (mirrors deploy.sh; no aws CLI needed)."""
import io, json, os, tarfile, time, urllib.request, zipfile, sys

import boto3

REGION = "us-east-2"
BUCKET = "the-sun-now"
FUNCTION = "sun-video-builder"
ROLE_NAME = "sun-video-builder-role"
LAYER_NAME = "ffmpeg-static"
RUNTIME = "python3.12"
HANDLER = "video_builder.handler.handler"
FFMPEG_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # this directory
ENV = {"VIDEO_FPS": "18", "FRAME_WINDOW": "144", "INTEGRATION_FRAMES": "3",
       "INTEGRATION_METHOD": "median", "SUN_BUCKET": BUCKET}

s = boto3.session.Session()
acct = s.client("sts").get_caller_identity()["Account"]
iam = s.client("iam")
lam = s.client("lambda", region_name=REGION)
s3 = s.client("s3", region_name=REGION)
role_arn = f"arn:aws:iam::{acct}:role/{ROLE_NAME}"


def step(m): print(f"\n=== {m} ===", flush=True)


# 1. IAM role
step("IAM role")
trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
try:
    iam.create_role(RoleName=ROLE_NAME,
                    AssumeRolePolicyDocument=json.dumps(trust),
                    Description="Sun-Right-Now video builder")
    iam.attach_role_policy(RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    print("created role; sleeping for IAM propagation"); time.sleep(12)
except iam.exceptions.EntityAlreadyExistsException:
    print("role exists")
s3pol = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:PutObjectAcl",
        "s3:DeleteObject"], "Resource": f"arn:aws:s3:::{BUCKET}/*"},
    {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{BUCKET}"}]}
iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="sun-bucket-access",
                    PolicyDocument=json.dumps(s3pol))
print("attached S3 policy")

# 2. ffmpeg layer
step("ffmpeg layer")
print("downloading static ffmpeg ...")
raw = urllib.request.urlopen(FFMPEG_URL, timeout=180).read()
print(f"downloaded {len(raw)/1e6:.1f} MB tarball")
ffmpeg_bytes = None
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as tf:
    for m in tf.getmembers():
        if m.name.endswith("/ffmpeg") and m.isfile():
            ffmpeg_bytes = tf.extractfile(m).read(); break
assert ffmpeg_bytes, "ffmpeg binary not found in tarball"
layer_zip = io.BytesIO()
with zipfile.ZipFile(layer_zip, "w", zipfile.ZIP_DEFLATED) as z:
    zi = zipfile.ZipInfo("bin/ffmpeg"); zi.external_attr = 0o755 << 16
    z.writestr(zi, ffmpeg_bytes)
data = layer_zip.getvalue()
print(f"layer zip {len(data)/1e6:.1f} MB; publishing ...")
if len(data) < 49_000_000:
    lv = lam.publish_layer_version(LayerName=LAYER_NAME,
        Description="static ffmpeg at /opt/bin/ffmpeg",
        Content={"ZipFile": data}, CompatibleRuntimes=[RUNTIME])
else:
    key = "deploy/ffmpeg-layer.zip"
    s3.put_object(Bucket=BUCKET, Key=key, Body=data)
    lv = lam.publish_layer_version(LayerName=LAYER_NAME,
        Description="static ffmpeg at /opt/bin/ffmpeg",
        Content={"S3Bucket": BUCKET, "S3Key": key}, CompatibleRuntimes=[RUNTIME])
layer_arn = lv["LayerVersionArn"]
print("published", layer_arn)

# 3. package code
step("package code")
code_zip = io.BytesIO()
with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ("__init__.py", "handler.py", "frame_queue.py", "manifest.py"):
        z.write(os.path.join(PKG_DIR, fn), f"video_builder/{fn}")
code = code_zip.getvalue()
print(f"code zip {len(code)} bytes")

# 4. create/update function (retry for IAM propagation)
step("lambda function")
common = dict(FunctionName=FUNCTION)
try:
    lam.get_function(**common)
    exists = True
except lam.exceptions.ResourceNotFoundException:
    exists = False

if exists:
    lam.update_function_code(FunctionName=FUNCTION, ZipFile=code)
    lam.get_waiter("function_updated").wait(FunctionName=FUNCTION)
    lam.update_function_configuration(FunctionName=FUNCTION, Handler=HANDLER,
        Runtime=RUNTIME, Role=role_arn, MemorySize=1024, Timeout=120,
        EphemeralStorage={"Size": 1024}, Environment={"Variables": ENV},
        Layers=[layer_arn])
    print("updated")
else:
    for attempt in range(6):
        try:
            lam.create_function(FunctionName=FUNCTION, Runtime=RUNTIME, Handler=HANDLER,
                Role=role_arn, MemorySize=1024, Timeout=120,
                EphemeralStorage={"Size": 1024}, Environment={"Variables": ENV},
                Layers=[layer_arn], Code={"ZipFile": code})
            print("created"); break
        except lam.exceptions.InvalidParameterValueException as e:
            print(f"  role not ready (attempt {attempt+1}): {str(e)[:80]}; retry in 8s")
            time.sleep(8)
    else:
        print("FAILED to create function"); sys.exit(1)
lam.get_waiter("function_updated").wait(FunctionName=FUNCTION)
func_arn = lam.get_function(FunctionName=FUNCTION)["Configuration"]["FunctionArn"]
print("function arn", func_arn)

# 5. S3 -> Lambda trigger
step("S3 trigger (prefix 1k/, suffix .png)")
try:
    lam.add_permission(FunctionName=FUNCTION, StatementId="s3invoke",
        Action="lambda:InvokeFunction", Principal="s3.amazonaws.com",
        SourceArn=f"arn:aws:s3:::{BUCKET}", SourceAccount=acct)
    print("added invoke permission")
except lam.exceptions.ResourceConflictException:
    print("invoke permission already present")
s3.put_bucket_notification_configuration(Bucket=BUCKET, NotificationConfiguration={
    "LambdaFunctionConfigurations": [{
        "LambdaFunctionArn": func_arn,
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {"Key": {"FilterRules": [
            {"Name": "prefix", "Value": "1k/"},
            {"Name": "suffix", "Value": ".png"}]}}}]})
print("bucket notification wired")
print("\nDONE.")

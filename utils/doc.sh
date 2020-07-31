#!/bin/bash
set -e # Exit if any errors occurr
CMD=$0
read -r -d '\0' DOCUMENTATION << EOF
Used for generating documentation for the project

Usage:
  $CMD generate - Runs pydoc over all modules
  $CMD deploy - Using aws-cli, deploys documentation to an S3 bucket
  $CMD prune - Using aws-cli, delete all buckets starting with vl-ml-doc- that do not have a matching branch on the remote repository.

Environment Variables:
  BUCKET_NAME - The name of the bucket to be deployed to (Default: current branch name)
\0
EOF

# Generates documentation for each module
function generate {
    pdoc3 --html --force covid
    pdoc3 --html --force discrete
    pdoc3 --html --force generate_discrete
    pdoc3 --html --force generate_master
    pdoc3 --html --force generate_ohe
    pdoc3 --html --force generate_predict
    pdoc3 --html --force ohe
    pdoc3 --html --force predict
}

# Deploys generated content to an AWS bucket
function deploy {
    if [ -z $BUCKET_NAME ]; then
	BUCKET_NAME=$(git branch | grep '*' | awk '{print $2}')
    fi
    BUCKET_NAME="vl-ml-doc-$(echo $BUCKET_NAME | sed 's/origin\///g')"
    BUCKET_NAME="$(echo $BUCKET_NAME | sed 's/[\/_]/-/g')"
    echo "Bucket name is $BUCKET_NAME"

    ## DO BUILD
    is_bucket_created="$(aws s3 ls | grep $BUCKET_NAME)" || true
    if [ ! -z "$is_bucket_created" ]; then
	echo "Bucket exists, removing old bucket."
	aws s3 rb s3://$BUCKET_NAME --force
    fi
    echo "Creating bucket."
    aws s3api create-bucket --acl public-read --bucket "$BUCKET_NAME" --region us-east-1
    echo "Setting bucket to be a static website."
    aws s3api put-bucket-website --bucket $BUCKET_NAME \
	--website-configuration "{\"IndexDocument\": {\"Suffix\": \"index.html\"}, \"ErrorDocument\": {\"Key\": \"index.html\"}}"

    echo "Copying html files."
    # Copies/syncs over the the compiled files making them public for the static website
    aws s3 sync html/ s3://$BUCKET_NAME --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
    echo "http://$BUCKET_NAME.s3-website-us-east-1.amazonaws.com/"
}

function prune {
  # Grab all buckets created by this script and format them to be the same as the branches
  BUCKET_LIST=$(aws s3 ls | grep vl-ml-doc | awk '{print $3}' | sed 's/vl-ml-doc-//g')

  # Grab all remote branches and format them
  git fetch -p
  BRANCHES_LIST=$(git branch -r | awk 'NR>1 {print $1}' | sed 's/origin\///g')

  for current_bucket in $BUCKET_LIST; do
    if [[ ! $BRANCHES_LIST =~ .*$current_bucket.* ]]; then
      # Format to match name in aws s3
      current_bucket="vl-ml-doc-$current_bucket"
      aws s3 rb s3://$current_bucket --force
    fi
  done
}


ACTION=$1

case "$ACTION" in
    generate) generate;;
    deploy) deploy;;
    prune) prune;;
  *)
  echo "$DOCUMENTATION"
  exit 1
esac

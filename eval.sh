sid="val9_sid.csv"
submit="luna_submission_res18_split_classify.csv"

#cp ${sid} ./evaluationScript/annotations/
#cp ${submit} ./evaluationScript/exampleFiles/submission/

#cd evaluationScript
mkdir ./evaluationScript_py3/exampleFiles/res18_split
python ./evaluationScript_py3/noduleCADEvaluationLUNA16.py \
        ./evaluationScript_py3/annotations/annotations.csv \
        ./evaluationScript_py3/annotations/annotations_excluded.csv \
        ${sid} \
        ${submit} \
        ./evaluationScript_py3/exampleFiles/res18_split/

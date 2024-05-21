import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  PutCommand,
} from "@aws-sdk/lib-dynamodb";

import fs from 'fs';
import { parseFasta } from '../cigartree.js';

const client = new DynamoDBClient({ region: process.env.AWS_DEFAULT_REGION });
const dynamo = DynamoDBDocumentClient.from(client);

if (process.argv.length != 4) {
    console.error('Usage: ' + process.argv[1] + ' id seqs.fa');
    process.exit(1);
}

const familyTableName = "evoldeeds-families";

const [ familyId, seqFilename ] = process.argv.slice(2);
const seqFile = fs.readFileSync(seqFilename).toString();
const { seqByName } = parseFasta (seqFile);
const lcSeqByName = Object.fromEntries (Object.keys(seqByName).map ((id) => [id, seqByName[id].toLowerCase().replaceAll('-','')]));

const create = async (id, seqById) => {
    await dynamo.send(
        new PutCommand({
        TableName: familyTableName,
        Item: {
            id,
            seqById
        },
        ConditionExpression: 'attribute_not_exists(id)'
        })
    );
    console.log(`Created ${id}`);
};

create (familyId, lcSeqByName);

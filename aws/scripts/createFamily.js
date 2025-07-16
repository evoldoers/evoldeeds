import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  PutCommand,
} from "@aws-sdk/lib-dynamodb";

import fs from 'fs';
import { parseFasta } from '../../js/cigartree.js';
import Getopt from 'node-getopt';

const awsDefaultRegion = 'us-east-1';
const client = new DynamoDBClient({ region: process.env.AWS_DEFAULT_REGION || awsDefaultRegion });
const dynamo = DynamoDBDocumentClient.from(client);

// Set up command line options
const getopt = new Getopt([
    ['k', 'keepcase', 'Keep original case of sequences (default: false)'],
    ['h', 'help', 'Display this help']
]).setHelp(
    'Usage: ' + process.argv[1] + ' [options] id seqs.fa\n' +
    'Options:\n' +
    '[[OPTIONS]]\n'
);

const opt = getopt.parse(process.argv.slice(2));

if (opt.options.help) {
    console.log(getopt.getHelp());
    process.exit(0);
}

if (opt.argv.length != 2) {
    console.error(getopt.getHelp());
    process.exit(1);
}

const familyTableName = "evoldeeds-families";

const [ familyId, seqFilename ] = opt.argv;
const seqFile = fs.readFileSync(seqFilename).toString();
const { seqByName } = parseFasta (seqFile, { forceLowerCase: !opt.options.keepcase, removeGaps: true });

const create = async (id, seqById) => {
    try {
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
    } catch (error) {
        console.error(`Error creating family '${id}':`, error.message);
        if (error.name === 'ConditionalCheckFailedException') {
            console.error(`Family '${id}' already exists in the database.`);
        } else if (error.name === 'ResourceNotFoundException') {
            console.error(`Table '${familyTableName}' was not found.`);
        }
        process.exit(1);
    }
};

create (familyId, seqByName);

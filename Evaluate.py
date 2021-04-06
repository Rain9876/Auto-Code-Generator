import random
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu
from matplotlib import pyplot as plt
import pickle
import re
import ast
import time
import torch
from Utils import *


# ========================================================================
# ========================================================================

set_rand_seeds()
device = set_device()

def RL_of_loss(model, x_in, y, mask, loss, tokenizer):
    w1 = 1
    w2 = 1

    model.eval()
    bz = x_in.size(0)

    with torch.no_grad():

        generated_ids = model.generate(
            input_ids=x_in,
            attention_mask=mask,
            num_beams=1,
            early_stopping=True
        )

        # preds = [tokenizer.decode(g) for g in generated_ids]
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        target = tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # print(preds)

    score1 = expression_evaluate(target, preds)

    score2 = logic_evaluate(target, preds)

    # Todo: Consider about this again

    null_loss = w1 * (1 - np.mean(score1)) + w2 * (1 - np.mean(score2)) + loss.item()

    # print("*******"*10)
    # print(bz)
    # print(score2)
    # print(score1)
    # print(loss.item())
    # print(null_loss)
    # print("*******"*10)

    loss_null = torch.tensor(null_loss).to(device=device)

    loss.data = loss_null.data

    return loss


def code_visitor(node, extractor, indent=-1):
    '''
        Recursively visit AST nodes with indent for a Module.
        Generally, only Statements, excepthandler, comprehension are in the consideration.
        Other types like expr, expr_context, boolop, operator, unaryop, cmpop, arguments are not.

        Following the AST document of python3: https://docs.python.org/3/library/ast.html#

        We consider to visit the logic behind the program. Therefore, it's trivial to visit
        low-level syntax node, such as Literals, Variables and the nodes inside Expression.

        We visit all Statement (stmt) nodes such as Assign, Raise, Delete, Pass, Break, Continue.,etc.
        Imports Statements, Control Flow (While, For, Try, With), Function and Class Defination:
        (FunctionDef, AsyncFunctionDef, ClassDef, Return, Global) are allowed as long as they are statements

        ExceptHandler and comprehension are categories that we considered as well. Some other sub-componets
        such as alias, arg, arguments are ignored.

        Args:
            node: AST parser root node
            extractor: Empty list to save extracted cmd-logic
            indent: the indent of each cmd-logic (1 indent = 4 space)

        Returns:
            extractor: a list saves all extracted cmd-logics and indents
    '''

    if isinstance(node, ast.stmt):
        extractor.append([type(node).__name__, indent])
    #         print(f"{indent} {type(node).__name__}")

    if isinstance(node, ast.excepthandler):
        indent -= 1
        extractor.append([type(node).__name__, indent])
    #         print(f"{indent} {type(node).__name__}")

    if sum(1 for x in ast.iter_child_nodes(node)) == 0:
        return

    else:

        # Special case for if condition
        # Causing ast syntax doesn't consider else/elif is a statement, but we do.
        # IF func is composed by test, body and orelse, while orelse can be empty or nested If.

        if "orelse" in node._fields and isinstance(node, ast.If) and len(node.orelse) > 0:

            code_visitor(node.test, extractor, indent)  # Visit test component (Actual only bool inside)

            indent += 1  # Visit body component with incremental indent
            for j in node.body:
                code_visitor(j, extractor, indent)

            indent -= 1  # A decremental indent for else
            extractor.append(["Else", indent])
            #             print(f"{indent} Else")

            indent += 1  # Visit orelse component with incremental indent
            for k in node.orelse:
                code_visitor(k, extractor, indent)

        else:

            indent += 1
            for i in ast.iter_child_nodes(node):
                code_visitor(i, extractor, indent)


def process_elif(extractor):
    """
        Handle the situation of Elif
        From the logic aspects, Elif == Else + if
    """
    i = 0
    while i < len(extractor) - 1:

        if extractor[i][0] == 'Elif':

            for k in range(i + 1, len(extractor)):
                # When Indent of next line logic is less than that of current elif
                # or equal but next line logic is Else or Elif.
                if extractor[i][1] < extractor[k][1] or \
                        extractor[i][1] == extractor[k][1] and extractor[k][0] in ["Else", "Elif"]:
                    extractor[k][1] += 1
                else:
                    break

            extractor[i][0] = "Else"
            extractor.insert(i + 1, ["If", extractor[i][1] + 1])

        i += 1

    return extractor


keywords = {"class": "ClassDef",
            "def": "FunctionDef",
            "for": "For",
            "while": "While",
            "if": "If",
            "with": "With",
            "try": "Try",
            "import": "Import",
            "except": "Excepthandler",
            "finally": "Finally",
            "else": "Else",
            "elif": "Elif"}


def recover_back(code):
    '''
        The special tokens of output code need to be replaced.
    '''
    code = re.sub(r"\s?ø\s?","\n", code)
    code = re.sub(r"§\s?","    ", code)
    return code


def replace_indent_newline(code):
    """
        Replace indent and newline with special symbol § and ø separately.
    """
    code = code.strip()
    lines = re.split(r'[\n;]', code)
    for i in range(len(lines)):
        lines[i] = re.sub("\s{4}", "§", lines[i])
    code = "ø".join(lines)
    return code


def num_of_indent(line):
    """
        Return the number of indent in the line.
    """
    indent = 0

    if not line or len(line) == 0:
        return 0

    while indent < len(line) and line[indent] == " ":
        indent += 1

    indent = indent // 4

    return indent


def remove_parentheses_newline_quotes(text):
    """
        Place all command in one-line, remove the multiple line style in string.
        Remove \n in parentheses, and remove contents in the quotes
    """
    stack = []
    quote = []
    status = False  # Indicates Whether parentheses and quotes are matched
    i = 0

    # Special Case: Remove s = “”“ multi-line string ”“”
    text = re.sub(r"(\"{3}|\'{3})[\s\S]*(\"{3}|\'{3})", "\"\"", text)

    while i < len(text):

        if text[i] in ["(", "{", "["] and len(quote) == 0:
            stack.append(i)

        elif text[i] in ["\'", "\""]:
            if len(quote) < 1:
                quote.append(i)

            elif len(quote) == 1:
                j = quote[0]
                if (text[j] == text[i]):
                    text = text[:j + 1] + text[i:]
                    quote.pop()
                    i = j + 1
                else:
                    quote.append(i)

            elif len(quote) > 1:
                j1 = quote[0]
                j2 = quote[1]

                if (text[j1] == text[i]):
                    text = text[:j1 + 1] + text[i:]
                    quote.pop()
                    quote.pop()
                    i = j1 + 1
                else:
                    text = text[:j2 + 1] + text[i:]
                    quote.pop()
                    i = j2 + 1

        elif text[i] in [")", "}", "]"] and len(quote) == 0:

            if len(stack) == 0:   # doesn't match! Only closed bracket
                i += 1
                continue

            j = stack.pop()

            if (text[j] == '(' and text[i] == ")") or \
                    (text[j] == '{' and text[i] == "}") or \
                    (text[j] == '[' and text[i] == "]"):

                if "\n" in text[j:i]:
                    tmp = re.sub(r"\n\s*", "", text[j:i])
                    text = text[:j] + tmp + text[i:]
                    i = j + len(tmp) + 1

        i += 1

    # Check Empty Stack
    if len(stack) == 0 and len(quote) == 0:
        status = True

    return text, status


def extractor_generated_code(code):
    extractor = []
    #     code = autopep8.fix_code(tmp, options={'aggressive': 1, 'ignore': ['W'], "jobs":4 })

    code, _ = remove_parentheses_newline_quotes(code)  # Remove newline in the parentheses of string
    code = re.sub(r'(?m)^\s*@.*?\n', '', code)  # Remove decorator like @staticmethod
    code = code.strip()

    list_code = re.split('\n|;', code)

    for i in list_code:

        indent = num_of_indent(i)

        try:
            code_visitor(ast.parse(i), indent - 1, extractor)

        except:

            i = i.rstrip(":")  # Remove symbol : at the end before split into words
            items = list(set(keywords.keys()) & set(i.split()))

            if len(items) > 0:
                kw = keywords.get(items.pop(0))
                extractor.append([kw, indent])
            #                 print(f"{indent} {kw}")

            else:
                extractor.append(["Error", indent])
    #                 print(f'{indent} {"Error"}')

    return extractor


def code_logic(text):
    '''
    Args:
        text: A string of code to be scanned.

    Returns:
        exec_: A Bool value to show whether the code is compilable under AST Parser.
        cmd_logics: A list of code logic with its indent.

    Minor-error exists

    '''

    cmd_logics = []

    try:
        #         text = autopep8.fix_code(text, options={'aggressive': 1, 'ignore': ['W']})
        #         text = remove_parentheses_newline_quotes(text)
        code = ast.parse(text)
        code_visitor(code, cmd_logics)
        exec_ = True

    except:

        # Parse cmd by cmd
        cmd_logics = process_elif(extractor_generated_code(text))
        exec_ = False

    return cmd_logics, exec_


###############################################################################################################
###############################################################################################################


# BLEU
def BLEU(reference, candidate, weights):
    '''
        Todo: Smooth function may required later.
    '''
    ref = reference.split()
    cand = candidate.split()
    score = sentence_bleu([ref], cand, weights=weights)
    return score


# Rouge
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def Rouge(output_lns, reference_lns):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


# PPL
def PPL(loss):
    pp = np.exp(loss)
    return pp


###############################################################################################################
###############################################################################################################


# Evaluation

def logic_evaluate(ref_code, hypo_code, sigma=0.5):
    '''
        There are two important criterions to evalute whether the logic of generated
        code is qualified. One is whether the code is executable, the other one is
        the control flow of the code, which includes indent and cmd_logics.

        We define the following formula to evalute the logic:

            logic_score = exec_ * sigma + BLEU * (1-sigma), where sigma can be 0.5.

        Given a higher logic score, the hypothesis code is closer to reference code.

        Considering the logic comparsion doesn't require semantic and contextual embedding,
        therefore, normal MT metrics is sufficent for evaluation. We are going to use BLUE,
        given more weights to n-gram, which n is larger than 1, such as bigram, tri-gram, etc.
        But there are some logic only have one or two combined cmds.

    '''

    logic_score = []
    assert len(ref_code) == len(hypo_code)

    for r, h in zip(ref_code, hypo_code):
        ref_logic, _ = code_logic(recover_back(r))  # Doesn't consider syntax error in ref
        hypo_logic, hypo_exec_ = code_logic(recover_back(h))

        #         ref = " ".join([f'{i[1]}_{i[0]}' for i in ref_logic])
        #         hypo = " ".join([f'{i[1]}_{i[0]}' for i in hypo_logic])

        ref = " ".join([f'{i[1]}_{i[0]}' for i in ref_logic])
        hypo = " ".join([f'{i[1]}_{i[0]}' for i in hypo_logic])

        score = BLEU(ref, hypo, (0.25, 0.25, 0.25, 0.25))

        logic_score.append(score)

    return np.array(logic_score)


def expression_evaluate(ref_code, hypo_code):
    '''
        Simply evaluate what the code express.
    '''
    metric_bleu = []

    for r, h in zip(ref_code, hypo_code):
        metric_bleu.append(BLEU(r, h, (0.25, 0.25, 0.25, 0.25)))

    return np.array(metric_bleu)




# from transformers import T5Tokenizer
# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# # print(tokenizer.get_vocab())
# tokenizer.add_special_tokens({'additional_special_tokens':["§","ø"]})
# print(tokenizer.vocab_size)

# ## ASCII code ["§", "ø"]
# print(chr(167))
# print(chr(248))

# input = 'Shows the face recognition results visually.\n\n    :param img_path: path to image to be recognized\n    :param predictions: results of the predict function\n    :return:'
#
# output =  "def show_prediction_labels_on_image(img_path, predictions):ø§pil_image = Image.open(img_path).convert('RGB')ø§draw = ImageDraw.Draw(pil_image)ø§for (name, (top, right, bottom, left)) in predictions:ø§§draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))ø§§name = name.encode('UTF-8')ø§§(text_width, text_height) = draw.textsize(name)ø§§draw.rectangle(((left, ((bottom - text_height) - 10)), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))ø§§draw.text(((left + 6), ((bottom - text_height) - 5)), name, fill=(255, 255, 255, 255))ø§del drawø§pil_image.show()"
#
# encode_input = tokenizer.encode(output)
# print(len([tokenizer.decode(i) for i in encode_input]))

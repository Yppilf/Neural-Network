#include "helper.ih"

string extractJsonField(const string &jsonStr, const string &fieldName) {
    size_t startPos = jsonStr.find("\"" + fieldName + "\":[");
    startPos += fieldName.length() + 4;     // +4 to account for "\":["
    size_t endPos = jsonStr.find("]", startPos);
    return jsonStr.substr(startPos, endPos-startPos);
}
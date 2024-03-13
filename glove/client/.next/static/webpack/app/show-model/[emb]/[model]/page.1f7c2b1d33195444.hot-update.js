"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("app/show-model/[emb]/[model]/page",{

/***/ "(app-pages-browser)/./src/apis/index.ts":
/*!***************************!*\
  !*** ./src/apis/index.ts ***!
  \***************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   GLOVE_CONFIG: function() { return /* binding */ GLOVE_CONFIG; },\n/* harmony export */   PHOBERT_CONFIG: function() { return /* binding */ PHOBERT_CONFIG; },\n/* harmony export */   \"default\": function() { return /* binding */ useAPI; }\n/* harmony export */ });\n/* harmony import */ var _swc_helpers_tagged_template_literal__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @swc/helpers/_/_tagged_template_literal */ \"(app-pages-browser)/./node_modules/@swc/helpers/esm/_tagged_template_literal.js\");\n/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! axios */ \"(app-pages-browser)/./node_modules/axios/lib/axios.js\");\n/* harmony import */ var _glove__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./glove */ \"(app-pages-browser)/./src/apis/glove.ts\");\n/* harmony import */ var _phobert__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./phobert */ \"(app-pages-browser)/./src/apis/phobert.ts\");\n/* provided dependency */ var process = __webpack_require__(/*! process */ \"(app-pages-browser)/./node_modules/next/dist/build/polyfills/process.js\");\n\nfunction _templateObject() {\n    const data = (0,_swc_helpers_tagged_template_literal__WEBPACK_IMPORTED_MODULE_0__._)([\n        \"\"\n    ]);\n    _templateObject = function() {\n        return data;\n    };\n    return data;\n}\n\n\n\nconst GLOVE_CONFIG = axios__WEBPACK_IMPORTED_MODULE_3__[\"default\"].create({\n    baseURL: process.env.SERVER_API + \"/glove\",\n    timeout: 10000\n});\nconst PHOBERT_CONFIG = axios__WEBPACK_IMPORTED_MODULE_3__[\"default\"].create({\n    baseURL: process.env.SERVER_API + \"/phobert\",\n    timeout: 10000\n});\nfunction useAPI(path) {\n    const emb = path[2];\n    const model = path[3];\n    if (emb == \"glove\") {\n        if (model == \"cnn\") return _glove__WEBPACK_IMPORTED_MODULE_1__.usingGLOVE.usingCNN;\n        else if (model == \"lstm\") return _glove__WEBPACK_IMPORTED_MODULE_1__.usingGLOVE.usingLSTM;\n        else if (model == \"bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_1__.usingGLOVE.usingBILSTM;\n        else if (model == \"ensemble_cnn_bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_1__.usingGLOVE.usingENSEMBLE_CNN_BILSTM;\n        else if (model == \"fusion_cnn_bilstm\") return _glove__WEBPACK_IMPORTED_MODULE_1__.usingGLOVE.usingFUSION_CNN_BILSTM;\n    } else if (emb == \"phobert\") {\n        if (model == \"cnn\") return _phobert__WEBPACK_IMPORTED_MODULE_2__.usingPHOBERT.usingCNN;\n        else if (model == \"lstm\") return _phobert__WEBPACK_IMPORTED_MODULE_2__.usingPHOBERT.usingLSTM;\n        else if (model == \"bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_2__.usingPHOBERT.usingBILSTM;\n        else if (model == \"ensemble_cnn_bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_2__.usingPHOBERT.usingENSEMBLE_CNN_BILSTM;\n        else if (model == \"fusion_cnn_bilstm\") return _phobert__WEBPACK_IMPORTED_MODULE_2__.usingPHOBERT.usingFUSION_CNN_BILSTM;\n    }\n    return None(_templateObject());\n}\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwcC1wYWdlcy1icm93c2VyKS8uL3NyYy9hcGlzL2luZGV4LnRzIiwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUF5QjtBQUNXO0FBQ0k7QUFFakMsTUFBTUcsZUFBZUgsNkNBQUtBLENBQUNJLE1BQU0sQ0FBQztJQUNyQ0MsU0FBU0MsT0FBT0EsQ0FBQ0MsR0FBRyxDQUFDQyxVQUFVLEdBQUc7SUFDbENDLFNBQVM7QUFDYixHQUFFO0FBRUssTUFBTUMsaUJBQWlCViw2Q0FBS0EsQ0FBQ0ksTUFBTSxDQUFDO0lBQ3ZDQyxTQUFTQyxPQUFPQSxDQUFDQyxHQUFHLENBQUNDLFVBQVUsR0FBRztJQUNsQ0MsU0FBUztBQUNiLEdBQUU7QUFFYSxTQUFTRSxPQUFPQyxJQUFZO0lBQ3ZDLE1BQU1DLE1BQU1ELElBQUksQ0FBQyxFQUFFO0lBQ25CLE1BQU1FLFFBQVFGLElBQUksQ0FBQyxFQUFFO0lBRXJCLElBQUlDLE9BQU8sU0FBUztRQUNoQixJQUFJQyxTQUFTLE9BQ1QsT0FBT2IsOENBQVVBLENBQUNjLFFBQVE7YUFDekIsSUFBSUQsU0FBUyxRQUNkLE9BQU9iLDhDQUFVQSxDQUFDZSxTQUFTO2FBQzFCLElBQUlGLFNBQVMsVUFDZCxPQUFPYiw4Q0FBVUEsQ0FBQ2dCLFdBQVc7YUFDNUIsSUFBSUgsU0FBUyx1QkFDZCxPQUFPYiw4Q0FBVUEsQ0FBQ2lCLHdCQUF3QjthQUN6QyxJQUFLSixTQUFTLHFCQUNmLE9BQU9iLDhDQUFVQSxDQUFDa0Isc0JBQXNCO0lBQ2hELE9BQ0ssSUFBSU4sT0FBTyxXQUFXO1FBQ3ZCLElBQUlDLFNBQVMsT0FDVCxPQUFPWixrREFBWUEsQ0FBQ2EsUUFBUTthQUMzQixJQUFJRCxTQUFTLFFBQ2QsT0FBT1osa0RBQVlBLENBQUNjLFNBQVM7YUFDNUIsSUFBSUYsU0FBUyxVQUNkLE9BQU9aLGtEQUFZQSxDQUFDZSxXQUFXO2FBQzlCLElBQUlILFNBQVMsdUJBQ2QsT0FBT1osa0RBQVlBLENBQUNnQix3QkFBd0I7YUFDM0MsSUFBS0osU0FBUyxxQkFDZixPQUFPWixrREFBWUEsQ0FBQ2lCLHNCQUFzQjtJQUNsRDtJQUVBLE9BQU9DO0FBRVgiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3JjL2FwaXMvaW5kZXgudHM/NGZkZiJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgYXhpb3MgZnJvbSAnYXhpb3MnXG5pbXBvcnQgeyB1c2luZ0dMT1ZFIH0gZnJvbSAnLi9nbG92ZSdcbmltcG9ydCB7IHVzaW5nUEhPQkVSVCB9IGZyb20gJy4vcGhvYmVydCdcblxuZXhwb3J0IGNvbnN0IEdMT1ZFX0NPTkZJRyA9IGF4aW9zLmNyZWF0ZSh7XG4gICAgYmFzZVVSTDogcHJvY2Vzcy5lbnYuU0VSVkVSX0FQSSArICcvZ2xvdmUnLFxuICAgIHRpbWVvdXQ6IDEwMDAwXG59KVxuXG5leHBvcnQgY29uc3QgUEhPQkVSVF9DT05GSUcgPSBheGlvcy5jcmVhdGUoe1xuICAgIGJhc2VVUkw6IHByb2Nlc3MuZW52LlNFUlZFUl9BUEkgKyAnL3Bob2JlcnQnLFxuICAgIHRpbWVvdXQ6IDEwMDAwXG59KVxuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiB1c2VBUEkocGF0aDogc3RyaW5nKSB7XG4gICAgY29uc3QgZW1iID0gcGF0aFsyXVxuICAgIGNvbnN0IG1vZGVsID0gcGF0aFszXVxuXG4gICAgaWYgKGVtYiA9PSAnZ2xvdmUnKSB7XG4gICAgICAgIGlmIChtb2RlbCA9PSAnY25uJykgXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdHTE9WRS51c2luZ0NOTlxuICAgICAgICBlbHNlIGlmIChtb2RlbCA9PSAnbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdHTE9WRS51c2luZ0xTVE1cbiAgICAgICAgZWxzZSBpZiAobW9kZWwgPT0gJ2JpbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdHTE9WRS51c2luZ0JJTFNUTVxuICAgICAgICBlbHNlIGlmIChtb2RlbCA9PSAnZW5zZW1ibGVfY25uX2JpbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdHTE9WRS51c2luZ0VOU0VNQkxFX0NOTl9CSUxTVE1cbiAgICAgICAgZWxzZSBpZiAoIG1vZGVsID09ICdmdXNpb25fY25uX2JpbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdHTE9WRS51c2luZ0ZVU0lPTl9DTk5fQklMU1RNXG4gICAgfVxuICAgIGVsc2UgaWYgKGVtYiA9PSAncGhvYmVydCcpIHtcbiAgICAgICAgaWYgKG1vZGVsID09ICdjbm4nKSBcbiAgICAgICAgICAgIHJldHVybiB1c2luZ1BIT0JFUlQudXNpbmdDTk5cbiAgICAgICAgZWxzZSBpZiAobW9kZWwgPT0gJ2xzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nUEhPQkVSVC51c2luZ0xTVE1cbiAgICAgICAgZWxzZSBpZiAobW9kZWwgPT0gJ2JpbHN0bScpXG4gICAgICAgICAgICByZXR1cm4gdXNpbmdQSE9CRVJULnVzaW5nQklMU1RNXG4gICAgICAgIGVsc2UgaWYgKG1vZGVsID09ICdlbnNlbWJsZV9jbm5fYmlsc3RtJylcbiAgICAgICAgICAgIHJldHVybiB1c2luZ1BIT0JFUlQudXNpbmdFTlNFTUJMRV9DTk5fQklMU1RNXG4gICAgICAgIGVsc2UgaWYgKCBtb2RlbCA9PSAnZnVzaW9uX2Nubl9iaWxzdG0nKVxuICAgICAgICAgICAgcmV0dXJuIHVzaW5nUEhPQkVSVC51c2luZ0ZVU0lPTl9DTk5fQklMU1RNXG4gICAgfVxuXG4gICAgcmV0dXJuIE5vbmVgYFxuXG59Il0sIm5hbWVzIjpbImF4aW9zIiwidXNpbmdHTE9WRSIsInVzaW5nUEhPQkVSVCIsIkdMT1ZFX0NPTkZJRyIsImNyZWF0ZSIsImJhc2VVUkwiLCJwcm9jZXNzIiwiZW52IiwiU0VSVkVSX0FQSSIsInRpbWVvdXQiLCJQSE9CRVJUX0NPTkZJRyIsInVzZUFQSSIsInBhdGgiLCJlbWIiLCJtb2RlbCIsInVzaW5nQ05OIiwidXNpbmdMU1RNIiwidXNpbmdCSUxTVE0iLCJ1c2luZ0VOU0VNQkxFX0NOTl9CSUxTVE0iLCJ1c2luZ0ZVU0lPTl9DTk5fQklMU1RNIiwiTm9uZSJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(app-pages-browser)/./src/apis/index.ts\n"));

/***/ }),

/***/ "(app-pages-browser)/./node_modules/@swc/helpers/esm/_tagged_template_literal.js":
/*!*******************************************************************!*\
  !*** ./node_modules/@swc/helpers/esm/_tagged_template_literal.js ***!
  \*******************************************************************/
/***/ (function(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   _: function() { return /* binding */ _tagged_template_literal; },\n/* harmony export */   _tagged_template_literal: function() { return /* binding */ _tagged_template_literal; }\n/* harmony export */ });\nfunction _tagged_template_literal(strings, raw) {\n    if (!raw) raw = strings.slice(0);\n\n    return Object.freeze(Object.defineProperties(strings, { raw: { value: Object.freeze(raw) } }));\n}\n\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwcC1wYWdlcy1icm93c2VyKS8uL25vZGVfbW9kdWxlcy9Ac3djL2hlbHBlcnMvZXNtL190YWdnZWRfdGVtcGxhdGVfbGl0ZXJhbC5qcyIsIm1hcHBpbmdzIjoiOzs7OztBQUFPO0FBQ1A7O0FBRUEsNERBQTRELE9BQU8sNkJBQTZCO0FBQ2hHO0FBQ3lDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9Ac3djL2hlbHBlcnMvZXNtL190YWdnZWRfdGVtcGxhdGVfbGl0ZXJhbC5qcz9kYjM3Il0sInNvdXJjZXNDb250ZW50IjpbImV4cG9ydCBmdW5jdGlvbiBfdGFnZ2VkX3RlbXBsYXRlX2xpdGVyYWwoc3RyaW5ncywgcmF3KSB7XG4gICAgaWYgKCFyYXcpIHJhdyA9IHN0cmluZ3Muc2xpY2UoMCk7XG5cbiAgICByZXR1cm4gT2JqZWN0LmZyZWV6ZShPYmplY3QuZGVmaW5lUHJvcGVydGllcyhzdHJpbmdzLCB7IHJhdzogeyB2YWx1ZTogT2JqZWN0LmZyZWV6ZShyYXcpIH0gfSkpO1xufVxuZXhwb3J0IHsgX3RhZ2dlZF90ZW1wbGF0ZV9saXRlcmFsIGFzIF8gfTtcbiJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(app-pages-browser)/./node_modules/@swc/helpers/esm/_tagged_template_literal.js\n"));

/***/ })

});
/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 * Some code taken from: https://github.com/jobergum/browser-ml-inference/tree/main
 * @format
 */

import React from 'react';
import {
  Image,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';
import {loadTokenizer} from './bert_tokenizer.ts';

import {
  Colors,
  DebugInstructions,
  Header,
  LearnMoreLinks,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';
import * as ort from 'onnxruntime-react-native';
import MNIST from './MNIST';
const {BertTokenizer} = require('bert-tokenizer');
const vocabUrl = 'node_modules/bert-tokenizer/assets/vocab.json';

var RNFS = require('react-native-fs');

function Section({children, title}) {
  const isDarkMode = useColorScheme() === 'dark';
  return (
    <View style={styles.sectionContainer}>
      <Text
        style={[
          styles.sectionTitle,
          {
            color: isDarkMode ? Colors.white : Colors.black,
          },
        ]}>
        {title}
      </Text>
      <Text
        style={[
          styles.sectionDescription,
          {
            color: isDarkMode ? Colors.light : Colors.dark,
          },
        ]}>
        {children}
      </Text>
    </View>
  );
}

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };
  const InferenceSession = ort.InferenceSession;
  const Tensor = ort.Tensor;

  function prepareDataA() {
    return Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  }
  function prepareDataB() {
    return Float32Array.from([
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
    ]);
  }

  function createRunOptions() {
    // run options: please refer to the other example for details usage for run options

    // specify log verbose to this inference run
    return {logSeverityLevel: 0};
  }

  const ai = async () => {
    // const Session = ort.InferenceSession;
    //   const NewSession = await Session.create('./roberta-sequence-classification-9.onnx');

    //   const inputs = {NewSession.inputNames[0].name: "hey"}
    //   NewSession.run(null, {NewSession.})
    // RNFS.readDir(RNFS.LibraryDirectoryPath)
    // .then(result => {
    //   console.log('GOT RESULT', result);
    //   result.map(resultNew => {
    //     RNFS.readDir(resultNew.path).then(result => {
    //       console.log(resultNew.path);
    //       result.map(resultNew => {
    //         console.log(resultNew.name);
    //       });
    //     });
    //     console.log(resultNew.name);
    //   });

    //   // stat the first file
    //   return Promise.all([RNFS.stat(result[0].path), result[0].path]);
    // })
    // .then(statResult => {
    //   if (statResult[0].isFile()) {
    //     // if we have a file, read it
    //     return RNFS.readFile(statResult[1], 'utf8');
    //   }

    //   return 'no file';
    // })
    // .then(contents => {
    //   // log the file contents
    //   console.log(contents);
    // })
    // .catch(err => {
    //   console.log(err.message, err.code);
    // });

    // RNFS.readFile('./app.json')
    //   .then(result => {
    //     console.log('GOT RESULT', result);
    //   })
    //   .catch(err => {
    //     console.log(err.message, err.code);
    //   });
    const modelPath = await MNIST.getLocalModelPathTwo();
    console.log({modelPath});
    try {
      // prepare inputs

      const session2 = await InferenceSession.create(modelPath);
      const tokenizer = loadTokenizer();

      function create_model_input(encoded, max = null) {
        var input_ids = new Array(max ? max + 2 : encoded.length + 2);
        var attention_mask = new Array(max ? max + 2 : encoded.length + 2);
        var token_type_ids = new Array(max ? max + 2 : encoded.length + 2);
        input_ids[0] = BigInt(101);
        attention_mask[0] = BigInt(1);
        token_type_ids[0] = BigInt(0);
        var i = 0;
        for (; i < encoded.length; i++) {
          input_ids[i + 1] = BigInt(encoded[i]);
          attention_mask[i + 1] = BigInt(1);
          token_type_ids[i + 1] = BigInt(0);
        }
        input_ids[i + 1] = BigInt(102);
        attention_mask[i + 1] = BigInt(1);
        token_type_ids[i + 1] = BigInt(0);
        i++;
        if (max) {
          for (; i < max + 1; i++) {
            input_ids[i + 1] = BigInt(0);
            attention_mask[i + 1] = BigInt(0);
            token_type_ids[i + 1] = BigInt(0);
          }
        }
        const sequence_length = input_ids.length;
        input_ids = new ort.Tensor('int64', BigInt64Array.from(input_ids), [
          1,
          sequence_length,
        ]);
        attention_mask = new ort.Tensor(
          'int64',
          BigInt64Array.from(attention_mask),
          [1, sequence_length],
        );
        token_type_ids = new ort.Tensor(
          'int64',
          BigInt64Array.from(token_type_ids),
          [1, sequence_length],
        );
        return {
          input_ids: input_ids,
          attention_mask: attention_mask,
          token_type_ids: token_type_ids,
        };
      }
      async function lm_inference(text) {
        try {
          // const encoded_ids = await tokenizer.then(t => {
          //   return t.tokenize(text);
          // });

          const encoded_ids = (await tokenizer).tokenize(text);
          console.log({encoded_ids});
          // const encoded_ids = [2023, 2003, 2019, 2742, 6251, 3231, 3231, 3231];
          const model_input = create_model_input(encoded_ids);
          console.log({
            input_ids: model_input.input_ids.data,
            attention_mask: model_input.attention_mask.data,
            token_type_ids: model_input.token_type_ids.data,
          });
          const output = await session2.run(model_input);

          console.log(output);

          // new Tensor().size

          // MEAN POOLING
          var input_mask_expaned_data = new Array(
            output.last_hidden_state.data.length,
          );
          for (var i = 0; i < model_input.attention_mask.data.length; i++) {
            for (var x = 0; x < 384; x++) {
              input_mask_expaned_data[x + i * 384] = Number(
                model_input.attention_mask.data[i],
              ).toExponential();
            }
          }
          // for (var i = 0; i < output.last_hidden_state.data.length; i++) {
          //   input_mask_expaned_data[i] = input_mask_expaned_data[x + i * 384] =
          //     Number(model_input.attention_mask.data.data[i]).toExponential();
          // }
          const input_mask_expanded = new Tensor(
            'float32',
            Float32Array.from(input_mask_expaned_data),
            [1, model_input.attention_mask.data.length, 384],
          );
          var inputMaskDataArray = [];
          for (var x = 0; x < 384; x++) {
            var value = 0;
            for (var i = 0; i < model_input.attention_mask.data.length; i++) {
              const index = () => (i === 0 ? x : i * 384 + x);
              value +=
                input_mask_expanded.data[index()] *
                output.last_hidden_state.data[index()];
            }
            inputMaskDataArray.push(value);
          }

          console.log({inputMaskDataArray});

          var clampInputMaskDataArray = [];
          for (var x = 0; x < 384; x++) {
            var value = 0;
            for (var i = 0; i < model_input.attention_mask.data.length; i++) {
              const index = () => (i === 0 ? x : i * 384 + x);
              value += input_mask_expanded.data[index()];
            }
            clampInputMaskDataArray.push(
              value >= 0.000000009 ? value : 0.000000009,
            );
          }
          console.log({clampInputMaskDataArray});
          for (var i = 0; i < inputMaskDataArray.length; i++) {
            inputMaskDataArray[i] =
              inputMaskDataArray[i] / clampInputMaskDataArray[i];
          }

          console.log({inputMaskDataArray});
          // END OF MEAN POOLING

          // NORMALIZE
          var normValue = 0;
          for (var i = 0; i < inputMaskDataArray.length; i++) {
            normValue += inputMaskDataArray[i] * inputMaskDataArray[i];
          }
          normValue = Math.sqrt(normValue);
          console.log({normValue});
          if (normValue < 0.000000000001) {
            normValue = 0.000000000001;
          }
          var normValueArray = [];
          for (var i = 0; i < inputMaskDataArray.length; i++) {
            normValueArray[i] = inputMaskDataArray[i] / normValue;
          }

          console.log({normValueArray});

          const finalValues = new Tensor(
            'float32',
            Float32Array.from(normValueArray),
            [1, 384],
          );

          // END OF NORMALIZE
          return finalValues;
        } catch (e) {
          console.error(e);
        }
        // if (
        //   (Array.isArray(text) && text.length === 1) ||
        //   !Array.isArray(text)
        // ) {

        // } else {
        //   const encoded_ids3 = [2023, 2003, 2019, 2742, 6251];
        //   const encoded_ids4 = [2023, 2003, 2019, 2742, 6251, 3231, 3231, 3231];
        //   const encoded_ids_arr = [encoded_ids3, encoded_ids4];
        //   try {
        //     var max = Math.max(...encoded_ids_arr.map(a => a.length));
        //     var model_inputs = [];
        //     console.log({max});

        //     //Model Outputs
        //     for (var i = 0; i < encoded_ids_arr.length; i++) {
        //       model_inputs.push(create_model_input(encoded_ids_arr[i], max));
        //     }
        //     var outputs = [];
        //     console.log({model_inputs: model_inputs[0].input_ids});
        //     console.log({model_inputs: model_inputs[1].input_ids});
        //     for (var i = 0; i < model_inputs.length; i++) {
        //       outputs.push(await session2.run(model_inputs[i]));
        //     }
        //     console.log('One', await session2.run(model_inputs[0]));
        //     console.log('Two', await session2.run(model_inputs[1]));

        //     //Mean Pooling
        //     var expandedMasks = [];
        //     var multi = [];
        //     for (
        //       var outputIndex = 0;
        //       outputIndex < outputs.length;
        //       outputIndex++
        //     ) {
        //       console.log(model_inputs[outputIndex].attention_mask.data);
        //       var input_mask_expaned_data = Array(
        //         model_inputs[outputIndex].attention_mask.data.length * 384,
        //       );
        //       for (
        //         var i = 0;
        //         i < model_inputs[outputIndex].attention_mask.data.length;
        //         i++
        //       ) {
        //         for (var x = 0; x < 384; x++) {
        //           input_mask_expaned_data[x + i * 384] = Number(
        //             model_inputs[outputIndex].attention_mask.data[i],
        //           ).toExponential();
        //         }
        //       }
        //       // for (var i = 0; i < outputs[1].last_hidden_state.data.length; i++) {
        //       //   input_mask_expaned_data[i] = Number(1).toExponential();
        //       // }
        //       console.log({input_mask_expaned_data});
        //       const input_mask_expanded = new Tensor(
        //         'float32',
        //         Float32Array.from(input_mask_expaned_data),
        //         [1, model_inputs[outputIndex].input_ids.data.length, 384],
        //       );
        //       expandedMasks.push(input_mask_expanded);
        //       var outputCopy = outputs[outputIndex].last_hidden_state;

        //       for (
        //         var i = 0;
        //         i < outputs[outputIndex].last_hidden_state.data.length;
        //         i++
        //       ) {
        //         outputCopy.data[i] =
        //           input_mask_expaned_data[i] *
        //           outputs[outputIndex].last_hidden_state.data[i];
        //       }

        //       var outputSum = [];
        //       for (var x = 0; x < 384; x++) {
        //         var value = 0;
        //         for (
        //           var i = 0;
        //           i < model_inputs[outputIndex].attention_mask.data.length;
        //           i++
        //         ) {
        //           const index = () => (i === 0 ? x : i * 384 + x);
        //           value += outputCopy.data[index()];
        //         }
        //         outputSum.push(value);
        //       }

        //       console.log({outputSum});
        //       multi.push(outputCopy);
        //     }
        //     console.log({expandedMasks});
        //     // console.log({input_mask_expanded});
        //     // var inputMaskDataArray = [];
        //     // for (var x = 0; x < 384; x++) {
        //     //   var value = 0;
        //     //   for (var i = 0; i < 7; i++) {
        //     //     const index = () => (i === 0 ? x : i * 384 + x);
        //     //     value +=
        //     //       input_mask_expanded.data[index()] *
        //     //       output.last_hidden_state.data[index()];
        //     //   }
        //     //   inputMaskDataArray.push(value);
        //     // }

        //     // console.log({inputMaskDataArray});
        //     //Normalization
        //   } catch (e) {
        //     console.error(e);
        //   }

        //   // for (var i = 0; i < encoded_ids_arr.length; i++) {
        //   //   model_inputs.push(create_model_input(encoded_ids_arr[i]));
        //   //   if (max > -1) {
        //   //     if (
        //   //       model_inputs[i].input_ids.length >
        //   //       model_inputs[max].input_ids.length
        //   //     ) {
        //   //       max = i;
        //   //     }
        //   //   } else {
        //   //     max = i;
        //   //   }
        //   // }
        //   // for (var i = 0; i < model_inputs.length; i++) {
        //   //   while (
        //   //     model_inputs[i].input_ids.length <
        //   //     model_inputs[max].input_ids.length
        //   //   ) {
        //   //     model_inputs[i].input_ids.push(0);
        //   //     model_inputs[i].attention_mask.push(0);
        //   //     model_inputs[i].token_type_ids.push(0);
        //   //   }
        //   // }
        //   console.log({model_inputs});
        // }
      }
      console.log(
        'lm_inference',
        await lm_inference('This is an example sentence'),
      );
      console.log(
        'lm_inference',
        await lm_inference('Each sentence is converted'),
      );
    } catch (e) {
      console.error(e);
    }
  };
  ai();

  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={backgroundStyle}></ScrollView>
      <View></View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
  },
  highlight: {
    fontWeight: '700',
  },
});

export default App;

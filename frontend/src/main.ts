import { createApp } from 'vue';
import {
  Button,
  Checkbox,
  ConfigProvider,
  Form,
  Input,
  Modal,
} from 'ant-design-vue';
import App from './App.vue';
import router from './routers';
import store from './stores';
import 'ant-design-vue/dist/reset.css';
import './styles/index.css';

const app = createApp(App);

app.use(store);
app.use(router);
app.use(ConfigProvider);
app.use(Button);
app.use(Checkbox);
app.use(Form);
app.use(Input);
app.use(Modal);
app.mount('#app');
